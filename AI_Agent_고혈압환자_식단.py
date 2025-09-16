# crew AI 를 사용하기 위해 langchain_openai가 필요 # 

!pip install langchain_openai
!pip install crewai
!pip install colab-xterm
%load_ext colabxterm
%xterm

# 아래를 한줄씩 마우스 오른쪽 붙여넣기로
# curl -fsSL https://ollama.com/install.sh | sh
# ollama serve & ollama pull exaone 3.5:2.4b

## 실습: 간단하게 에이전트와 TASK를 설정하여 만들어보기
## 스토리 부여: 고혈압 환자의 식단 레시피 짜기

from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import os

# 크류 AI는 Open API 키가 있어야 한다. 그래서 NA라는 Fake를 줄 것이다.
os.environ["OPENAI_API_KEY"] = "NA"

# 에이전트로 사용할 LLM 정의
llm = ChatOpenAI(
    model="ollama/exaone3.5:2.4b", #ollama pull exaone3.5:2.4b 사용
    base_url="http://localhost:11434"  # 내 컴퓨터 11434 포트 (올라마 서버)에 있음
)

# 콘텐츠 기획자 에이전트=역할 정의(3개를 만든다.)
# 콘텐츠 기획자 에이전트 정의
planner = Agent(
    role="콘텐츠 기획자",
    goal="주제 {topic}에 대해 흥미롭고 사실에 기반한 콘텐츠를 기획한다. 모든 결과는 한국어로 작성합니다.",
    backstory="당신은 질병 관련된 레시피 책에서 주제 {topic}에 관한 글 기획 작업을 진행 중입니다. "
              "독자들이 유익한 식단 정보를 얻고 올바른 결정을 내릴 수 있도록 도와주는 건강정보를 수집합니다. "
              "자세한 개요와 관련 주제, 하위 주제들을 준비하세요. 이 작업은 콘텐츠 작가가 글을 작성하는 기반이 됩니다. "
              "답변은 반드시 한국어로 작성됩니다.",
    llm=llm,
    allow_delegation=False,
    verbose=True
)

# 콘텐츠 작가 에이전트 정의 = 의견기고문
writer = Agent(
    role="콘텐츠 작가",
    goal="주제 {topic}에 대한 통찰력 있고 사실에 기반한 의견 기고문을 작성한다. 모든 결과는 한국어로 작성합니다.",
    backstory="당신은 주제 {topic}에 관한 새로운 의견 기고문을 작성 중입니다. "
              "콘텐츠 기획자가 제공한 개요와 관련 정보를 기반으로 글을 작성하세요. "
              "개요의 주요 목표와 방향을 따르며, 객관적이고 공정한 통찰을 제공하고, 출처를 명확히 하세요. "
              "의견과 객관적 사실을 구분해야 합니다. 답변은 반드시 한국어로 작성됩니다.",
    llm=llm,
    allow_delegation=False,
    verbose=True
)

# 편집자 에이전트 정의 = 작성된 글을 편집/작성
editor = Agent(
    role="편집자",
    goal="주어진 블로그 글을 보건복지부에서 권고하는 글쓰기 스타일에 맞게 편집한다. 모든 결과는 한국어로 작성합니다.",
    backstory="당신은 영양관리 콘텐츠 잡지 작가로부터 전달받은 글을 검토하는 편집자입니다. "
              "블로그 글이 언론의 최선의 관행을 따르고, 의사의 의견이나 영양관리사 등 전문가의 의견과 근거 문헌에 기반하여 균형 잡힌 시각을 제공하며, "
              "논란의 여지가 있는 주제나 의견은 피하도록 교정하세요. 결과는 반드시 꼭 한국어로 작성됩니다.",
    llm=llm,
    allow_delegation=False,
    verbose=True
)

# 콘텐츠 기획 태스크 정의 = 작업수행 해야하는 작업들에 대해 기술한다.
plan = Task(
    description=(
        "1. 주제 {topic}와 관련된 최신 트렌드, 일상생활에서 쉽게 정용가능한, 주목할 만한 뉴스를 우선 파악합니다.\n"
        "2. 대상 독자를 분석하여, 그들의 주요 관심사와 문제점을 고려합니다.\n"
        "3. 서론, 핵심 포인트, 행동 촉구(Call to Action)를 포함한 상세 콘텐츠 개요를 작성합니다.\n"
        "4. SEO 키워드와 관련 데이터, 출처를 포함합니다."
    ),
    expected_output="개요, 독자 분석, SEO 키워드 및 참고 자료가 포함된 포괄적인 콘텐츠 문서",
    agent=planner,
)

# 콘텐츠 작성 태스크 정의
write = Task(
    description=(
        "1. 콘텐츠 기획 문서를 바탕으로 주제 {topic}에 관한 유익한 글을 작성합니다.\n"
        "2. SEO 키워드를 자연스럽게 포함합니다.\n"
        "3. 각 섹션 및 부제목을 매력적으로 구성합니다.\n"
        "4. 흥미로운 서론, 통찰력 있는 본문, 그리고 쉽게 이해할 수 있는 결론으로 구성된 글의 구조를 갖춥니다.\n"
        "5. 문법 오류 및 브랜드의 목소리에 맞게 교정합니다.\n"
    ),
    expected_output="출판 준비가 된, 마크다운 형식의 잘 작성된 글 (각 섹션은 2~3 단락 포함)",
    agent=writer,
)

# 블로그 글 편집 태스크 정의
edit = Task(
    description="주어진 근거문서의 글을 친근하고, 이해하기 쉬운 일상용어로 변경하여 기술합니다.",
    expected_output="고혈압 환자에게 가장 중요한 것은 꾸준한 식단 관리와 운동입니다.",
    agent=editor
)

# 에이전트와 태스크를 포함하는 크루 생성
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=True
)

# 입력값 설정 및 크루 실행
inputs = {"topic": "고혈압 환자의 건강관리를 위한 식단"}
result = crew.kickoff(inputs=inputs)


final_answer = result.dict()['raw']
print(final_answer)

    ### self feedback: [부족한 점]
    # 샘플 이미지가 없음 (식단의 샘플을 이미지로 보여주면 더 좋을 것 같다.)
    # 연령대를 고려한 상세한 설명이 필요하다.
    # tarket 고객이 누구인지를 고려해야 한다.
    #   예를 들어,
    #   고혈압 환자는 50대 이상인 사람이 많을 텐대, 그렇다면 현미, 통밀빵이라고 제시하기 보다, 
    #   현미밥(쌀과 현미를 몇대 몇으로 밥을 지어야 하는지, 통밀 빵은 무엇을 말하는지 함유량 등을 직접적으로 제시하는 등) 으로 확실한 예시를 든다. 
    # 연어도 고등어나 삼치와 같은 것으로 대체 할 만한 것들을 추가 제시 해 준다.
    # 콩류도 예시를 들어 준다.
    # 식단을 직접 참고할 수 있는 사이트를 제시할 수 있다.
