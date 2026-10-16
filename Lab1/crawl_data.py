import pandas as pd
import requests
from bs4 import BeautifulSoup

url = "https://www.formula1.com/en/results/2026/drivers"
headers = { "User-Agent": ( "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML," " like Gecko) Chrome/120.0.0.0 Safari/537.36")}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")
table = soup.find("table")
headers = []
for th in table.find_all("th"):
    text = th.text.strip()
    if text:
        headers.append(text)
data = []
rows = table.find_all("tr")[1:] 
for tr in rows:
    cols = tr.find_all("td")
    if cols:
        noidung_row = []
        for td in cols:
            text = td.text.strip() 
            if text != "":
                if len(text) > 3 and text[-3:].isupper() and text[-4].islower():
                    text = text[:-3].strip() 
                noidung_row.append(text)
        if len(noidung_row) >= len(headers):
            clean_row = noidung_row[: len(headers)]
            data.append(clean_row)

df = pd.DataFrame(data, columns=headers)
df.to_csv("f1_drivers_2026.csv", index=False, encoding="utf-8-sig")
print("File da luu")