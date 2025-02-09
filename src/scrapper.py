import httpx
import asyncio
from bs4 import BeautifulSoup
from langchain.document_loaders import UnstructuredHTMLLoader

async def fetch_html(url):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text

if __name__ == "__main__":
    url = "https://hbr.org/2023/01/what-makes-a-great-resume"
    html = asyncio.run(fetch_html(url))
    soup = BeautifulSoup(html, "html.parser")
    loader = UnstructuredHTMLLoader()
    main_content = loader.load(html)
    print(html)