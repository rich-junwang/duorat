import gdown
import sys

gdrive_id = sys.argv[1]
output_file = sys.argv[2]

url = f"https://drive.google.com/uc?export=download&id={gdrive_id}"
#url = f"https://drive.google.com/file/d/{gdrive_id}/view?usp=sharing"
#url = f"https://drive.google.com/u/0/open?id={gdrive_id}"
print(url)

gdown.download(url, output_file, quiet=False)
