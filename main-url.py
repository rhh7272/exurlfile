# pip install --upgrade langchain langchain-community langchain-text-splitters langchain-openai langchain-chroma python-dotenv youtube-transcript-api pytube pypdf docx2txt unstructured openpyxl python-pptx beautifulsoup4

import os  # 내부에 들어가 있는 코어 라이브러리 (설치 없이 사용 가능)
import tempfile
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import (
    YoutubeLoader,
    WebBaseLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.title("📄/🎥 멀티미디어 & 문서 질문 및 요약 봇")

# 대화 기록을 저장할 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def load_url(url):
    if "youtube.com" in url or "youtu.be" in url:
        try:
            # 우선 영상 정보(제목, 작성자 등)와 함께 로드 시도
            loader = YoutubeLoader.from_youtube_url(
                url, 
                add_video_info=True, 
                language=["ko", "en", "en-US", "ja", "zh", "es", "fr", "de"]
            )
            docs = loader.load()
        except Exception as e:
            # pytube 오류 발생 시 영상 정보 없이 자막만 로드 시도 (Fallback)
            st.toast("영상 메타데이터를 가져오는 데 실패하여 자막만 추출합니다.")
            try:
                loader = YoutubeLoader.from_youtube_url(
                    url, 
                    add_video_info=False, 
                    language=["ko", "en", "en-US", "ja", "zh", "es", "fr", "de"]
                )
                docs = loader.load()
            except Exception:
                docs = []
    else:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
        except Exception as e:
            st.error(f"웹페이지를 불러오는 중 오류가 발생했습니다: {e}")
            docs = []
    return docs

def load_document(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_file_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(tmp_file_path)
        elif uploaded_file.name.endswith(".xlsx"):
            loader = UnstructuredExcelLoader(tmp_file_path)
        elif uploaded_file.name.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(tmp_file_path)
        else:
            raise ValueError("지원하지 않는 파일 형식입니다.")
        
        docs = loader.load()
    finally:
        try:
            os.unlink(tmp_file_path)
        except Exception:
            pass
    return docs

def init_rag_chain(docs):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,           # 하나의 청크가 가질 최대 글자 수
        chunk_overlap  = 20,        # 청크 간 문맥 연결을 위해 겹칠 글자 수
        length_function = len,      # 길이 측정 기준 (기본 문자열 길이)
        is_separator_regex = False, # 구분 기호의 정규표현식 해석 여부
    )
    texts = text_splitter.split_documents(docs)

    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    db = Chroma.from_documents(texts, embeddings_model)

    # 대화형 LLM 모델 초기화 (예: GPT-4o-mini, 디폴트는 무료인 gpt-3.5-turbo)
    # 멀티 쿼리 리트리버 생성 및 LLM과 설정, 모델을 명시적으로 지정할 수 있음 (예: gpt-4o-mini)
    llma = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # temperature=0은 답변의 일관성을 높이기 위해 사용 (2로 세팅하면 창의성이 증가)

    # 사용자의 질문을 다양한 각도에서 재해석해서 검색 확률을 높이는 멀티 쿼리 리트리버(MultiQueryRetriever) 개체를 생성
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever = db.as_retriever(), 
        llm = llma
    )

    # RAG 체인 생성
    # LLM(시스템) 프롬프트 정의: LLM이 질문에 답할 때 사용할 지침과 맥락을 포함
    system_prompt = (
        "너는 질문-답변을 돕는 유능한 비서야. "
        "아래 제공된 맥락(context)만을 사용하여 질문에 답해줘. "
        "답을 모르면 모른다고 하고, 절대 답변을 지어내지 마.\n\n"
        "{context}"
    )
    # 대화형 프롬프트 템플릿 생성: 시스템(system) 프롬프트와 사용자(human or user) 입력을 결합하여 LLM이 이해할 수 있는 형식으로 구성
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 검색된 문서를 활용하여 질문에 답변하는 체인 생성
    question_answer_chain = create_stuff_documents_chain(llma, prompt)

    # RAG 체인 생성: 멀티 쿼리 리트리버와 질문-답변 체인을 결합하여, 사용자의 질문에 대해 검색된 문서를 활용하여 답변을 생성하는 체인
    rag_chain = create_retrieval_chain(retriever_from_llm, question_answer_chain)
    
    return rag_chain

input_type = st.radio("입력 방식을 선택하세요:", ["웹/유튜브 URL", "문서 파일 업로드 (PDF, Word, Excel, PPT)"])

docs_loaded = False
content_title = "알 수 없는 제목"
content_source = ""

if input_type == "웹/유튜브 URL":
    # URL 입력란
    input_url = st.text_input("웹페이지 또는 유튜브 영상 URL을 입력해주세요:", placeholder="https://...")
    
    if input_url:
        # 새로운 URL이 입력되었거나 아직 처리되지 않은 경우
        if "current_source" not in st.session_state or st.session_state.current_source != input_url:
            with st.spinner("URL 내용을 추출하고 분석 중입니다. 잠시만 기다려주세요..."):
                docs = load_url(input_url)
                if not docs:
                    st.error("내용을 추출할 수 없습니다. (지원하지 않는 URL이거나 차단된 페이지입니다)")
                    st.stop()
                
                st.session_state.docs = docs
                st.session_state.rag_chain = init_rag_chain(docs)
                st.session_state.current_source = input_url
                st.session_state.summary = None # 새로운 URL 입력 시 요약 초기화
                st.session_state.chat_history = [] # 새로운 URL 입력 시 대화 기록 초기화
                
                st.success("URL 내용 분석 완료!")
                
        docs_loaded = True
        docs = st.session_state.docs
        content_title = docs[0].metadata.get('title', '알 수 없는 제목')
        if "youtube.com" in input_url or "youtu.be" in input_url:
            content_source = f"채널: {docs[0].metadata.get('author', '알 수 없는 작성자')}"
        else:
            content_source = f"출처: {docs[0].metadata.get('source', input_url)}"

elif input_type == "문서 파일 업로드 (PDF, Word, Excel, PPT)":
    uploaded_file = st.file_uploader("분석할 문서를 업로드해주세요", type=['pdf', 'docx', 'xlsx', 'pptx'])
    
    if uploaded_file is not None:
        if "current_source" not in st.session_state or st.session_state.current_source != uploaded_file.name:
            with st.spinner(f"'{uploaded_file.name}' 문서를 분석 중입니다. 잠시만 기다려주세요..."):
                docs = load_document(uploaded_file)
                if not docs:
                    st.error("문서 내용을 추출할 수 없습니다.")
                    st.stop()
                
                st.session_state.docs = docs
                st.session_state.rag_chain = init_rag_chain(docs)
                st.session_state.current_source = uploaded_file.name
                st.session_state.summary = None
                st.session_state.chat_history = []
                
                st.success("문서 분석 완료!")
                
        docs_loaded = True
        docs = st.session_state.docs
        content_title = uploaded_file.name
        content_source = "업로드된 문서"

if docs_loaded:
    st.subheader(f"📄 {content_title}")
    st.caption(content_source)

    # 요약 버튼 추가
    if st.button("📝 요약하기", type="secondary"):
        with st.spinner("AI가 내용을 요약하는 중입니다..."):
            llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
            prompt_template = """다음 제공된 내용을 바탕으로 핵심 내용을 상세하게 요약해 주세요.
반드시 **한국어(Korean)**로 자연스럽게 번역하여 작성해 주세요.
가독성을 높이기 위해 글머리 기호(Bullet points)를 사용해 주세요.

내용:
{text}

한국어 요약:"""
            prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
            chain = prompt | llm | StrOutputParser()
            combined_text = "\n\n".join([doc.page_content for doc in docs])
            st.session_state.summary = chain.invoke({"text": combined_text})

    # 요약 결과 출력
    if st.session_state.get("summary"):
        st.markdown("### 📝 요약 내용")
        st.info(st.session_state.summary)

    st.divider()

    # 질문 입력란
    question = st.text_input("질문을 입력하세요:", placeholder="예: 이 영상의 핵심 내용은 무엇인가요?")

    # 버튼을 나란히 배치하기 위해 컬럼 사용
    col1, col2 = st.columns([1, 1])
    with col1:
        ask_btn = st.button("답변 생성", type="primary")
    with col2:
        clear_btn = st.button("전체내용삭제", type="secondary")

    # 내용 삭제 버튼 동작
    if clear_btn:
        st.session_state.chat_history = []
        st.rerun() # 화면 갱신

    # 답변 생성 버튼 동작
    if ask_btn:
        if question.strip():  # 질문이 비어 있지 않을 경우
            with st.spinner("답변을 생각하는 중입니다..."):
                response = st.session_state.rag_chain.invoke({"input": question})
                
                # 최신 대화가 가장 위에 오도록 리스트의 맨 앞(index 0)에 삽입
                st.session_state.chat_history.insert(0, {"question": question, "answer": response['answer']})
        else:
            st.warning("먼저 질문을 입력해 주세요.")
            
    # 대화 기록 출력
    if st.session_state.chat_history:
        st.subheader("💬 대화 기록")
        for chat in st.session_state.chat_history:
            st.info(f"**🗣️ 질문:** {chat['question']}")
            st.write(f"**🤖 답변:** {chat['answer']}")
            st.divider()
else:
    st.info("웹페이지/YouTube URL을 입력하거나 분석할 문서를 업로드해 주세요.")
    if "rag_chain" in st.session_state:
        del st.session_state["rag_chain"]
    if "current_source" in st.session_state:
        del st.session_state["current_source"]
    if "docs" in st.session_state:
        del st.session_state["docs"]
    if "summary" in st.session_state:
        del st.session_state["summary"]
    if "chat_history" in st.session_state:
        del st.session_state["chat_history"]
