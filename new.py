elif selected_page== "Stock News":
    
    st.markdown("""
        <style>
            .title {
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("<h1 class='title'>Stock News</h1>", unsafe_allow_html=True)
    st.markdown("""

### Overview

Stay updated with the latest news related to your favorite stocks. Our Stock News feature allows you to get the latest articles and updates on any stock you are interested in.

### Instructions

1. **Enter the Stock Name:**
    - In the input box, type the name of the stock for which you want to get the news.
    - Example: To get news about Google, you would enter `Google`.

2. **Submit:**
    - After entering the stock name, press the 'Enter' key or click outside the input box to submit.

3. **View News:**
    - The latest news articles related to the entered stock will be displayed below.


                """)
    
     # Add content for video summary page
    # Now we have To get Youtube Link from the user
    stock_name = st.text_input("Enter Stock Name")
    if st.button("Enter"):
        if stock_name:
            try:
                  # Combine transcript text
                  with st.spinner('Please wait...'):
                    response = gen_ai_model.generate_content("""Generate a comprehensive news summary about the stock: """ + stock_name + """
                    Please format the response properly in markdown format to display on my website with the following structure:
                    1. Stock Name: Display the name of the stock.
                    2. Latest News: Provide a summary of the latest news articles related to the stock, using proper headings and subheadings.
                    - Headline: Provide the headline of the news article.
                    - Source: Mention the source of the news article.
                    - Date: Provide the date of the news article.
                    - Summary: Provide a brief summary of the news article.
                    - Link: Provide a link to the full news article.
                    The stock is: """+stock_name)
                    text = response._result.candidates[0].content.parts[0].text
                  st.markdown(text)
           
            except Exception as e:
             st.error(f"An error occurred: {e}")

    
    
    st.markdown("""
                Stay informed with real-time news updates and make well-informed investment decisions.
                """)