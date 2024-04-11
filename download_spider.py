import os
import gdown  # pyright: ignore[reportMissingTypeStubs]
import zipfile

def download_spider_zip(output_path: str = "spider.zip"):
  assert output_path.endswith(".zip")
  spider_url = "https://drive.google.com/u/0/uc?id=1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m&export=download"  # https://yale-lily.github.io/spider
  if not os.path.exists(output_path):
    gdown.download(spider_url, output_path, quiet=False) # pyright: ignore[reportUnknownMemberType]
  return output_path

def extract_zip(zip_path: str, output_directory: str = "."):
  assert zip_path.endswith(".zip")
  with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(output_directory)
  return zip_path.removesuffix(".zip")

if __name__ == "__main__":
  download_spider_zip(output_path="spider.zip")
  extract_zip("spider.zip", output_directory=".")