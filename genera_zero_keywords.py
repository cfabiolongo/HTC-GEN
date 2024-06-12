# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional
import fire
from llama import Llama, Dialog
import pandas as pd





def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Scrittura file scores
    output = "results/results_total_keywords_generation.txt"
    with open(output, "w") as file:
        file.write("ZERO-SHOT total keywords generation:\n\n")

    # Definisci le colonne del DataFrame
    colonne = ['Y1', 'Domain', 'Y', 'area', 'keyword']
    # Crea un DataFrame vuoto con le colonne definite
    df_base = pd.DataFrame(columns=colonne)
    # taxonomy file
    taxonomy = 'dataset/wos_taxonomy.xlsx'
    # Leggi il file Excel e crea un DataFrame
    df = pd.read_excel(taxonomy)

    for index, row in df.iterrows():
        sys_content = "Consider only keywords, in a numbered list, only made of more than one words (without description), in English"
        user_content = "Generate the most 20 used keywords, only different from "+row['area'].strip()+", in the scientific area: "+row['area'].strip()+" ("+row['Domain'].strip()+")"
        print("\n\nElaborazione n. "+str(index)+" - area: "+row['area'].strip()+" - domain: "+row['Domain'].strip())

        dialogs: List[Dialog] = [
            [
                {
                    "role": "system",
                    "content": sys_content,
                },
                {"role": "user", "content": user_content},
            ]
        ]
        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for dialog, result in zip(dialogs, results):
            for msg in dialog:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            print(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")

            # append testo generato
            with open(output, "a") as file:
                file.write(str(result['generation']['content']).strip())
                file.write("\n==================================\n")

            # Dividi il documento in paragrafi utilizzando le righe vuote come separatore
            keys_raw = str(result['generation']['content']).strip().split('\n\n')
            keys = keys_raw[-1]

            # Crea un nuovo DataFrame con le stringhe estratte
            df_nuovo = pd.DataFrame({'Y1': row['Y1'], 'Domain': row['Domain'], 'Y': row['Y'], 'area': row['area'], 'keyword': keys}, index=[0])

            # Concatena il nuovo DataFrame con il DataFrame vuoto df_base
            df_base = pd.concat([df_base, df_nuovo])

            print("\n==================================\n")

        df_base.to_excel("dataset/zeroshot_total_keywords_wos.xlsx")

if __name__ == "__main__":
    fire.Fire(main)
