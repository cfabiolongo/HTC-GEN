from transformers import RobertaModel, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
import pandas as pd
from tqdm import tqdm

def text_to_embedding(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs.to(device))
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def cosine_distance(embedding1, embedding2):
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)
    return 1.0 - cosine_similarity(embedding1, embedding2)[0, 0]


def fix_abstract(row):
    abstract = str(row['Abstract'])
    return abstract

def main():

    num_classes = 133
    classe_label = 'Y'
    testo_label = 'Abstract'
    
    file_name = "dataset/wos_train.xlsx"
    print(f"Nome file: {file_name}\n")

    # Leggi i dati dal file Excel in un DataFrame pandas
    df = pd.read_excel(file_name)
   

    # Carica il modello e il tokenizer di XML-RoBERTa-Large
    model = RobertaModel.from_pretrained("xlm-roberta-large")
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')

    # Impostazione del dispositivo di calcolo (CPU o GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)

    min_cosine_distances = []
    mean_cosine_distances = []
    max_cosine_distances = []

    for i in range(num_classes):

        print("Elaborazione Classe: ",i)

        df_area = df[df[classe_label] == i]

        df_area[testo_label] = df_area.apply(fix_abstract, axis=1)    
        texts = df_area[testo_label].tolist()    

        embeddings = [text_to_embedding(text, model, tokenizer, device) for text in texts]
        combinations_list = list(combinations(embeddings, 2))

        cosine_distances = []
        for emb1, emb2 in tqdm(combinations_list, desc="Calcolo Distanza Coseno"):
            cosine_distances.append(cosine_distance(emb1, emb2))

        min_cosine = min(cosine_distances)
        mean_cosine = sum(cosine_distances) / len(cosine_distances)
        max_cosine = max(cosine_distances)   

        print(f"\nCosine Similarity classe {i}\nMinima: {min_cosine}\nMedia: {mean_cosine}\nMassima: {max_cosine}\n")

        min_cosine_distances.append(min_cosine)
        mean_cosine_distances.append(mean_cosine)
        max_cosine_distances.append(max_cosine)


    average_mean = sum(mean_cosine_distances) / len(mean_cosine_distances)
    average_min = sum(min_cosine_distances) / len(min_cosine_distances)
  
    print(f"\n\nChamfer Distance Score: {average_min}\nRemote Clique Score: {average_mean}")
    



if __name__ == "__main__":
    main()