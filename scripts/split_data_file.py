import os
import pandas as pd

def split_file(input_file, chunk_size_mb=50, output_dir="data/data_split_files"):
    os.makedirs(output_dir, exist_ok=True)
    
    chunk_size = chunk_size_mb * 1024 * 1024 
    file_num = 1
    current_size = 0
    chunk_path = os.path.join(output_dir, f"data_split_file_{file_num}.txt")
    
    with open(input_file, 'r', encoding="utf-8") as infile:
        header = infile.readline() 
        
        out = open(chunk_path, 'w', encoding="utf-8")
        out.write(header) 
        
        for line in infile:
            line_size = len(line.encode("utf-8"))
            
            if current_size + line_size > chunk_size:
                out.close()
                file_num += 1
                chunk_path = os.path.join(output_dir, f"data_split_file_{file_num}.txt")
                out = open(chunk_path, 'w', encoding="utf-8")
                current_size = 0 
            
            out.write(line)
            current_size += line_size
        
        out.close()

    print(f"Splitting complete: {file_num} chunks created in '{output_dir}'.")

def read_chunks_into_dataframe(output_dir="data/data_split_files"):
    chunk_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith("data_split_file_")])

    df_list = [pd.read_csv(chunk_files[0], sep="\t", header=0)]
    
    for chunk in chunk_files[1:]:
        print("Appending next chunk")
        df_list.append(pd.read_csv(chunk, sep="\t", header=None, names=df_list[0].columns))
    
    final_df = pd.concat(df_list, ignore_index=True)
    
    return final_df

input_file = "data/bbs50-can_naturecounts_filtered_data.txt"  
split_file(input_file)
df = read_chunks_into_dataframe()
print(df.columns)
print(df.head()) 