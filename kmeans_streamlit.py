import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.cluster import KMeans
from skimage import color
from PIL import Image, ImageDraw, ImageFont
from streamlit_drawable_canvas import st_canvas

import warnings
warnings.filterwarnings('ignore')

if 'id' not in st.session_state:
    st.session_state.id = 0

def main():
    st.title("K-means Clustering")
    sel = st.sidebar.radio('Color space:', ('RGB', 'LAB'))
    fup = st.sidebar.file_uploader('Upload image file')
    filename = fup.name if fup else './color_chart.jpg'
    img = Image.open(filename)
    # st.image(img)

    MAX_iter = 30
    resize = 4096
    N_CLUSTERS = st.sidebar.selectbox('Number of clusters:', list(range(2, 11)), index=3)
    mk_color = st.sidebar.radio('3D chart color:', ('gradation', 'color'))
    sel_mark = st.sidebar.radio('COG marker:', ('off', 'on'))
    sel_mode = st.sidebar.radio('Select:', ('All', 'Region'))
    print(f'Session state: {st.session_state.id}')

    if sel_mode == 'Region':
        sk_color = st.sidebar.radio('Stroke color:', ('black', 'white'))
        dr_mode  = st.sidebar.radio('Drawing mode:', ('rect', 'circle'))
        img2, ra = get_region(img, sk_color, dr_mode)
        img3, rb = img_resize(img2, ra, resize)
        st.write('Selected region')
        st.image(img_cut(img2, dr_mode))
        rgb = np.array(img3)
        drgb = rgb.reshape((-1, 3)) if sel=='RGB' else hex2lab(rgb).reshape((-1, 3))
        dset = dset_cut(drgb, rgb, dr_mode, rb)
    else:
        img3, rb = img_resize(img, img.width, resize)
        st.image(img)
        rgb = np.array(img3)
        dset = rgb.reshape((-1, 3)) if sel=='RGB' else hex2lab(rgb).reshape((-1, 3))

    cls = KMeans(n_clusters = N_CLUSTERS, max_iter = MAX_iter)
    try:
        idx = cls.fit_predict(dset)
        cog = cls.cluster_centers_
    except (ValueError):
        st.write(':red[Region failure]')
        return

    ratio, idx2 = color_ratio(dset, idx, cog, N_CLUSTERS, sel)
    plot_graph(dset, idx, idx2, cog, ratio, N_CLUSTERS, sel, mk_color, sel_mark)
    plot_color(ratio, N_CLUSTERS)

    file_dload('3D_chart')
    file_dload('cluster_colors')
    if sel_mode == 'Region':
        file_dload('region_image')
    
    st.session_state.id += 1
    return

def get_region(img, sk_color, mode):
    color = '#000000' if sk_color=='black' else '#ffffff'
    canvas_result = st_canvas(
        fill_color = None,          # 塗りつぶしの色
        stroke_width = 1,           # 線の太さ
        stroke_color = color,       # 線の色
        drawing_mode = mode,        # 描画モード
        background_color = None,    # 背景色
        background_image = img,     # 背景画像
        width = img.width,          # キャンバスの幅
        height = img.height,        # キャンバスの高さ
        # update_streamlit = True,  # Streamlitをリアルタイムで更新
        # key = "canvas",           # キャンバスのキー
    )
    try:
        if canvas_result.json_data['objects'] != []:
            p = canvas_result.json_data['objects']
            i = len(p)-1 if len(p) > 0 else 0
            (xt, yt, xw, yh, ph) = (p[i]['left'], p[i]['top'], p[i]['width'], p[i]['height'], p[i]['angle'])
        else:
            img2 = img.resize((int(img.width/5), int(img.height/5)))
            cr = int(img.width/2) if img.width <= img.height else int(img.height/2)
            return img2, cr
    except (TypeError):
        img2 = img.resize((int(img.width/5), int(img.height/5)))
        cr = int(img.width/2) if img.width <= img.height else int(img.height/2)
        return img2, cr       

    if mode == 'circle':
        ax = xt + xw/2.0*np.cos(ph*np.pi/180.0)
        ay = yt + yh/2.0*np.sin(ph*np.pi/180.0)
        (cx, cy, cr) = (int(ax), int(ay), int(xw/2.0))
        print(f'Region {i+1}: center ({cx}, {cy}), radius = {cr}')
        img2 = img.crop((cx-cr, cy-cr, cx+cr, cy+cr))
    else:
        (x0, y0, x1, y1) = (xt, yt, xt+xw, yt+yh)
        print(f'Region {i+1}: ({x0}, {y0}) - ({x1}, {y1})')
        img2 = img.crop((x0, y0, x1, y1))
        cr = 0
    return img2, cr

def img_cut(img, mode):
    if mode == 'circle':
        img2 = Image.new("RGB", img.size, (255, 255, 255))
        mask = Image.new("1", img.size, 1)
        draw = ImageDraw.Draw(mask)
        (cx, cy) = (int(img.width/2), int(img.height/2))
        cr = cy - 1 if cx > cy else cx - 1
        draw.ellipse((cx-cr, cy-cr, cx+cr, cy+cr), fill=0)
        img_o = Image.composite(img2, img, mask)
    else:
        img_o = img
    img_o.save('./out_region_image.png')
    return img_o

def dset_cut(dset, rgb, mode, r):
    if mode == 'circle':
        pos = [True for i in dset]
        col = int(len(dset)/len(rgb))
        for i in range(len(dset)):
            if (i%col - int(col/2))**2 + (int(i/col) - int(col/2))**2 > r**2:
                pos[i] = False
        dset_o = [s for i, s in enumerate(dset) if pos[i]]
    else:
        dset_o = dset
    # print(f'ratio = {len(dset_o)/len(dset):.2f}')
    return dset_o

def img_resize(img, r0, pix):
    (w0, h0) = img.size
    if w0*h0 > pix:
        sc = np.sqrt(w0*h0/pix)
        (w1, h1, ra) = (int(w0/sc), int(h0/sc), int(r0/sc))
        img2 = img.resize((w1, h1))
    else:
        img2 = img
        ra = r0
    return img2, ra

def color_ratio(dset, idx, cog, n, sel):
    ndata = len(dset)
    count = [0 for i in range(n)]
    for i in range(ndata):
        count[idx[i]] += 1

    ratio = []
    for i in range(n):
        ratio.append((rgb2hex(cog[i] if sel=='RGB' else lab2rgb(cog[i])), count[i]/ndata*100, i))
    ratio_s = sorted(ratio, key=lambda x:x[1], reverse=True)

    rvs = [0 for i in range(n)]
    for i in range(n):
        rvs[ratio_s[i][2]] = i
    idx2 = np.array([rvs[idx[i]] for i in range(ndata)])

    for i in range(n):
        s = f'{hex2rgb(ratio_s[i][0])}' + ' '*10
        print(f'{i+1}:', f'{ratio_s[i][0]}  ', s[0:16], f'{ratio_s[i][1]:.2f}%')
    print()
    return ratio_s, idx2

def plot_graph(dset, idx, idx2, cog, ro, n, sel1, sel2, sel3):
    item = ['R', 'G', 'B', 'cluster', 'Cluster', 'mark', 'opacity'] if sel1=='RGB' \
        else ['L*', 'a*', 'b*', 'cluster', 'Cluster', 'mark', 'opacity']

    df1 = pd.DataFrame(dset)
    df1['3'] = idx+1  if sel2=='gradation' else [str(i+1) for i in idx]
    df1['4'] = idx2+1 if sel2=='gradation' else [str(i+1) for i in idx2]
    df1['5'] = [0.1 for i in dset]
    df1['6'] = [0.5 for i in dset]
    df1.columns = item

    df2 = pd.DataFrame(cog.astype(int))
    df2['3'] = [i+1 for i in range(n)]  if sel2=='gradation' else [str(i+1) for i in range(n)]
    df2['4'] = [ro[i][2]+1 for i in range(n)]  if sel2=='gradation' else [str(ro[i][2]+1) for i in range(n)]
    df2['5'] = [1 for i in cog]
    df2['6'] = [1.0 for i in cog]
    df2.columns = item

    df = pd.concat([df1, df2])
    df.sort_values('Cluster', ascending=True, inplace=True)
    colors = [ro[i][0] for i in range(n)]
    # pd.set_option('display.max_rows', None)
    # print(df)
    
    fig = px.scatter_3d(df, x='R', y='G', z='B', color='Cluster', color_discrete_sequence=colors, size='mark', opacity=1.0) if sel1=='RGB' \
        else px.scatter_3d(df, x='a*', y='b*', z='L*', color='Cluster', color_discrete_sequence=colors, size='mark', opacity=1.0)

    fig.update_layout(title='3D Chart', width=500, height=500)
    if sel3 == 'off':
        fig.update_traces(marker_size=1)
    st.plotly_chart(fig, use_container_width=True)
    pio.write_image(fig, './out_3D_chart.png', format='png')
    return

def plot_color(ratio, n):
    bgcolor   = '#ffffff'
    textcolor = '#000000'
    textsize  = 20
    img  = Image.new('RGB', (500, (n+1)*75), bgcolor)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", size=textsize)

    for i in range(n):
        rectcolor = ratio[i][0]
        ypos = i*75 + 75
        draw.rectangle([(50, ypos-15), (100, ypos+35)], outline=rectcolor, fill=rectcolor, width=0)
        draw.text(( 15, ypos), f'{i+1}',                fill=textcolor, font=font)
        draw.text((125, ypos), f'{rectcolor}',          fill=textcolor, font=font)
        draw.text((230, ypos), f'{hex2rgb(rectcolor)}', fill=textcolor, font=font)
        draw.text((400, ypos), f'{ratio[i][1]:.1f} %',  fill=textcolor, font=font)

    st.subheader('Results')
    st.image(img)
    img.save('./out_cluster_colors.png')
    return

def file_dload(fname):
    with open('./out_'+fname+'.png', 'rb') as file:
        img = file.read()

    st.sidebar.download_button(
        label = fname,
        data = img,
        file_name = 'out_'+fname+'.png',
        mime = 'image/png'
    )
    return

def hex2rgb(a):
    return (int(a[1:3], 16), int(a[3:5], 16), int(a[5:7], 16))

def rgb2hex(a):
    return f'#{int(a[0]):02x}' + f'{int(a[1]):02x}' + f'{int(a[2]):02x}'

def lab2rgb(a):
    return [int(c) for c in color.lab2rgb(a)*255]

def hex2lab(rgb):
    lab = []
    for c in rgb:
        lab.append(color.rgb2lab(c/255.0))
    return np.array(lab)

if __name__ == '__main__':
    main()
