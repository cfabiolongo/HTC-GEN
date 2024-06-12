# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional
import fire
from llama import Llama, Dialog
import pandas as pd


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 1.5,
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
    output = "results/results_abstract_daArea100-p1_generation.txt"
    with open(output, "w") as file:
        file.write("ZERO-SHOT abstract from area generation:\n\n")

    # Definisci le colonne del DataFrame
    colonne = ['Y1', 'Domain', 'Y', 'area']
    # Crea un DataFrame vuoto con le colonne definite
    df_base = pd.DataFrame(columns=colonne)
    # taxonomy
    taxonomy = 'dataset/conteggi_aree_total_keywords.xlsx'
    df = pd.read_excel(taxonomy)

    for index, row in df.iterrows():
        sys_content = "Consider only abstracts including keywords (separated by commas), in English"
        user_content = "Generate an abstract of scientific paper, in the area: "+row['area'].strip()+" ("+row['Domain'].strip()+")"

        print("\n\nElaborazione n. "+str(index)+" - area: "+row['area'].strip()+" - domain: "+row['Domain'].strip())
        for x in range(0, 100):
            print("\n ----> Passaggio n: "+str(x)+" di", row['area'].strip())

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

                # Crea un base nuovo DataFrame con tutti i codice in base alle iterazioni
                df_nuovo = pd.DataFrame({'Y1': row['Y1'], 'Domain': row['Domain'], 'Y': row['Y'], 'area': row['area']}, index=[0])

                # Concatena il nuovo DataFrame con il DataFrame vuoto df_base
                df_base = pd.concat([df_base, df_nuovo])

                print("\n==================================\n")

        excel_gen = "zeroshot_daArea100-p1_wos.xlsx"
        df_base.to_excel(excel_gen)

if __name__ == "__main__":
    fire.Fire(main)
