import json
from bs4 import BeautifulSoup

# ===========================
# 1. JSON 파일 목록 및 청킹 설정
# ===========================
json_files = ["parsed_report1.json", "parsed_report2.json", "parsed_report3.json"]
chunk_size = 500
all_chunks = []

# ===========================
# 2. 각 JSON 파일 처리
# ===========================
for file in json_files:
    print(f"Processing {file} ...")
    with open(file, "r", encoding="utf-8") as f:
        parsed_data = json.load(f)

    # HTML 추출
    html_content = parsed_data.get("content", {}).get("html", "")
    if not html_content:
        print(f"{file}에 HTML 없음, 스킵")
        continue

    soup = BeautifulSoup(html_content, "html.parser")

    # ===========================
    # 3. 문단 텍스트 추출
    # ===========================
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    # 제목(h1~h6)도 포함
    headings = [h.get_text(strip=True) for h in soup.find_all(["h1","h2","h3","h4","h5","h6"])]

    # ===========================
    # 4. 표(table) 텍스트 추출
    # ===========================
    tables_text = []
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            row_text = [td.get_text(strip=True) for td in tr.find_all(["td","th"])]
            if row_text:
                tables_text.append(" | ".join(row_text))  # 셀 구분은 '|' 사용

    # ===========================
    # 5. 모든 텍스트 합치기 (청킹용)
    # ===========================
    all_text_lines = headings + paragraphs + tables_text

    # ===========================
    # 6. 청킹
    # ===========================
    current_chunk = []
    for line in all_text_lines:
        words = line.split()
        while words:
            remaining_space = chunk_size - len(current_chunk)
            current_chunk.extend(words[:remaining_space])
            words = words[remaining_space:]
            if len(current_chunk) >= chunk_size:
                all_chunks.append(" ".join(current_chunk))
                current_chunk = []
    if current_chunk:
        all_chunks.append(" ".join(current_chunk))

print(f"총 {len(all_chunks)}개의 청크 생성 완료")

# ===========================
# 7. 청크 저장
# ===========================
with open("rag_chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print("RAG용 청크를 'rag_chunks.json'로 저장 완료")
