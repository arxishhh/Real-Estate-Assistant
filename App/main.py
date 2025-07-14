import streamlit as st
from rag import process_urls,generate_answer
st.title('Real Estate Research Tool')
url1 = st.sidebar.text_input('URL 1')
url2 = st.sidebar.text_input('URL 2')
url3 = st.sidebar.text_input('URL 3')

process_url_button = st.sidebar.button('Start Research')
placeholder = st.empty()
if process_url_button:
    urls = [url for url in (url1, url2, url3) if len(url) != 0]
    if len(urls) == 0:
        placeholder.warning('You must provide at least one URL')
    else:
        for status in process_urls(urls):
            placeholder.text(status)

query = placeholder.text_input('Please Ask Your Question')
if query:
    try :
        answer, sources = generate_answer(query)
        st.header('Answer:')
        st.write(answer)
        if sources!= '':
            st.subheader("Sources:")
            for source in sources.split('\n'):
                st.write(source)
    except RuntimeError as e:
       placeholder.text('Please Enter The Urls Before Asking Your Question')
