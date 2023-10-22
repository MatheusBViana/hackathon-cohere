import streamlit as st
import pandas as pd
import pandas as pd
import cohere
from sklearn.decomposition import PCA
import numpy as np

import cohere
import requests
from PIL import Image

st.title("Image2Playlist")

imagem1 = {
    "url": "https://images.pexels.com/photos/2034851/pexels-photo-2034851.jpeg?cs=srgb&dl=pexels-edoardo-tommasini-2034851.jpg&fm=jpg",
    "description": "nightclub, party, disco"
}

imagem2 = {
    "url": "https://images.fineartamerica.com/images/artworkimages/mediumlarge/2/friends-having-fun-on-beach-tom-kelley-archive.jpg",
    "description": "beach, surf, vacation, summer"
}

imagem3 = {
    "url": "https://f.i.uol.com.br/fotografia/2021/10/18/1634577429616dac156d431_1634577429_3x2_md.jpg",
    "description": "coffee, cafe, photography, restaurant"
}

imagem4 = {
    "url": "https://img.freepik.com/fotos-premium/fundo-de-paisagem-calma-vibrante-linda-cachoeira-de-corrego-na-floresta-tropical-tempo-do-nascer-do-sol-verao-manha-efeito-de-brilho-do-sol-natureza-tranquila-relaxar-viagem-ferias-papel-de-parede_270304-1040.jpg",
    "description": "nature, calming, serene, scenic, fresh"
}


image_dict = {
    "Imagem 1": imagem1,
    "Imagem 2": imagem2,
    "Imagem 3": imagem3,
    "Imagem 4": imagem4
}



col_left, col_right = st.columns(2)

# ----- Análise de correlação por gênero (heatmap) -----
with col_left:
    st.markdown("Geração de playlist a partir de uma imagem")
    option = st.selectbox(
    'Escolha a imagem a ser analisada:',
    ("Imagem 1", "Imagem 2", "Imagem 3", "Imagem 4"))
    st.markdown("""
    A imagem selecionada gerará um prompt de texto para ser utilizado pela API.
    """) 
    generate_button = st.button("Generate list of musics")

with col_right:
    st.image(image_dict[option]["url"], caption=option, use_column_width=True)
    
# Specify the path to your CSV file and the column name with text data
csv_file_path = 'train.csv'
text_column_name = 'Lyrics'  # Replace with the actual column name

# Initialize the Cohere client with your API key
co = cohere.Client('*')

# Function to embed text data
def embed_text(text_data, model='small'):
    response = co.embed(texts=text_data, model=model)
    return response


df = pd.read_csv(csv_file_path, nrows=3000, on_bad_lines='error')

if generate_button:

    st.write("Descrição da imagem:", image_dict[option]["description"])

    text_prompt = image_dict[option]["description"]

    def read_csv_and_embed(csv_file_path, text_column_name):
        global df
        text_data = df[text_column_name].tolist()

        text_data.append(text_prompt)
        embeddings = embed_text(text_data)
        return embeddings


    # Call the function to read the CSV and embed the text data
    embeddings = read_csv_and_embed(csv_file_path, text_column_name)
    df = pd.read_csv(csv_file_path, nrows=15000)
    novo_df=df.drop(columns=['Lyrics'])
    novo_df.to_pickle('train_novo.csv')


    mask = df['Lyrics'].str.len() < 6000
    df = df[mask]
    artist_data = df['Artist'].tolist()
    name_data = df['Song'].tolist()

    text_data = df[text_column_name].tolist()
    
            
    lyrics_embedding = pd.read_pickle("embed.pkl").values.tolist()
    prompt_embedding = np.vstack(embed_text([text_prompt]))
        
    # Apply PCA to reduce the dimensionality
    pca = PCA(n_components=500)  # Specify the number of components you want
    lyrics_embedding.append(prompt_embedding[0])
        
    reduced_embeddings = pca.fit_transform(lyrics_embedding)

        
    reduced_embeddings=reduced_embeddings.tolist()

    prompt=reduced_embeddings[-1]
    prompt=np.array(prompt)
    reduced_embeddings.pop()
    mindist = 100000
    i=0
    min_id=-1
    dist_list=[]
    for emb in reduced_embeddings:
        dist = np.linalg.norm(np.array(emb)-prompt)
        if i >=0 and i<14900:
            dist_list.append([dist,name_data[i],i,artist_data[i]])
        i+=1
    dist_list=sorted(dist_list, key=lambda x: x[0])
    songs=[]
    for d in dist_list[0:200]:
        d[0]=round(d[0],2)
        songs.append(text_data[d[2]])


    unique_list=[]
    for item in songs:
        if item not in unique_list:
            unique_list.append(item)

    query = "What are the lyrics with mood most similar to '"+text_prompt+"'"
    print(query)
    results = co.rerank(query=query, documents=unique_list, top_n=10, model='rerank-english-v2.0') # Change top_n to change the number of results returned. If top_n is not passed, all results will be returned.
    st.header("Músicas mais parecidas:")
    for idx, r in enumerate(results):
        st.write(f"Song: {dist_list[r.index][1]} - {dist_list[r.index][3]}")



