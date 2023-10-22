
import os
#import dotenv
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objs as go
import plotly.express as px


def create_horizontal_bar_chart(data, title):
    fig = px.bar(data, x="Percentage", y="Skill", orientation="h")
    fig.update_layout(title_text=title, showlegend=False)
    return fig

def display_project_details(project): #use a dict to hold all projects and then
    data_dict = {
      'Project 1: Finetuning ChatGPT via Azure OpenAI':['Successfully led a transformative project focused on finetuning the ChatGPT model via Azure OpenAI modules to train a powerful internal Larg e Language Model (LLM). \
      This LLM was specifically d esigned to extract valuable insights for our company, revolutionizing our decision making processes.'],
       'Project 2: Integrating ChatGPT with Azure Cognitive Search and Langchain':['Creating a vector search via Azure Cognitive Search for in context learning of ChatGPT model for enterprise level solution and adding \
       Langchain for semantic search and retrieval to create a high performing , finetuned ChatGPT for organization level usage with minimalistic hallucinations'],
       'Project 3: Movie Recommendation System on Big Data':['Injecting and processing data into MS SQL studio via python and ORM architecture of SQLAlchemy Connecting the database remotely via Streamlit API to create web app \
        on MovieLens 25 million ratings dataset.', \
        "images/mov1.PNG", "images/mov2.PNG", "images/mov3.PNG"],
       'Project 4: Electric Vehicle Mobility detection':['Classify state of EV based on yearly data for several models and types of EV using data from emobpy and ACN and identifying data drift.', \
       "images/ev1.JPG", "images/ev2.JPG", "images/ev3.JPG"],
       'Project 5: UnConstrained optimization using PSO':['Implemented solution for un constrained optimizations using particle swarm algorithm.'],
       'Project 6: Object detection using transfer learning':['Created Traffic sign detection and classificat ion model using Inception ResnetV2 and VGG16', \
       "images/obj1.PNG", "images/obj2.PNG", "images/obj3.PNG"],
       'Project 7: Feature Selection via Boruta':['Implemented Boruta paper for feature selection based on entropy of decision tree classifiers and regressors.'],
       'Project 8: Explainable AI (XAI) and Palantir Foundry':['Developing and improving explainable AI module for PwC to improve customer trust in artificial intelligence and machine learning based solution to explain \
        how different drivers impact the decision. Also developed and deployed AI & ML solutions via Palantir Foundry platform specialized for time series and forecasting problems.'],
       'Project 9: Predicting Defaulters Based on Previous Payments':['A classical ML classification project done on Microsoft Azure, where I built two solutions one via AutoML and other with finetuning tree based models\
       to precisely predict defaulters based on previous payment patterns. I ensured fair ML practice by elimination algorithmic bias.', "images/az1.PNG", "images/az2.PNG", "images/az3.PNG"],
       'Project 10: Human Activity Recognition(HAR)':["Completed this project as part of IBM's Advance Data Science Capstone, I used deeplearning to recognize human activity via sensor data", \
        "images/har1.PNG", "images/har2.PNG", "images/har3.PNG"],
       'Project 11: Analyzing Spatial Business Opportunites via Follium':['Part of IBM Data Science certification, I created this dataset on my own using public data from Govt. of Paikstan and using Follium API to understand \
       trends and happenings in different areas of city, combined with spatial analysis to suggest prominent business opportunites in the city.', \
       "images/fol1.PNG", "images/fol2.PNG", "images/fol3.PNG"],
       'Project 12: Soccer Anyalsis':['A fun project to test some Visualization skills with my favorite game', "images/soc1.PNG", "images/soc2.PNG", "images/soc3.PNG"]

    }
    chart_dict = {
      'Project 1: Finetuning ChatGPT via Azure OpenAI':[["Python", "Machine Learning", "NLP", "Microsoft Azure", "OpenAI", "Data preprocessing", "Ethics and Bias Consideration"], [90, 80, 80, 100, 80, 90, 90]],
       'Project 2: Integrating ChatGPT with Azure Cognitive Search and Langchain':[["Azure Cognitive Search", "LangChain", "Vector & semantic search",  "Python", "API Integration", "JSON", "Project Management"], \
       [100, 85, 85, 80, 80, 80, 80]],
       'Project 3: Movie Recommendation System on Big Data':[["PySpark", "SQLAlchemy", "RDBMS", "Machine Learning", "Dashboard", "Matrix Factorization", "Feature Engineering", "Scalability and Performance", \
       "Parallel Computing", "Data Mining"], [90, 85, 90, 90, 90,80, 85, 90, 80, 80]],
       'Project 4: Electric Vehicle Mobility detection':[["Python", "Machine Learning", "Predictive Modeling", "Statistical Analysis", "Time Series Analysis"], [80, 80, 80, 90, 90]],
       'Project 5: UnConstrained optimization using PSO':[["Python", "Algorithm Design and Analysis", "Numerical Optimization", "Fitness Function Design", "Convergence Analysis"], [85, 90, 85, 100, 100]],
       'Project 6: Object detection using transfer learning':[["Computer Vision", "Deep Learning", "Data Annotation", "Transfer Learning", "Model Fine-Tuning"], [90, 80, 75, 90, 90]],
       'Project 7: Feature Selection via Boruta':[["Python", "Statistical Significance Testing", "Model Interpretability", "Hyperparameter Tuning", "Boruta Algorithm"], [70, 80, 90, 80, 80]],
       'Project 8: Explainable AI (XAI) and Palantir Foundry':[["Python", "R", "Explainable AI (XAI)", "Palantir Foundry", "Model Explanation Techniques", "Ethical AI"], [80, 80, 90, 80, 75, 70]],
       'Project 9: Predicting Defaulters Based on Previous Payments':[["Python", "Machine Learning", "Data Analysis", "Feature Engineering", "Imbalanced Data Handling"], [70, 80, 70, 75,80]],
       'Project 10: Human Activity Recognition(HAR)':[["Python", "Deep Learning", "Statistical Analysis", "Time Series Analysis"], [80, 80, 85, 80]],
       'Project 11: Analyzing Spatial Business Opportunites via Follium':[["Python", "Follium", "Machine Learning", "Geospatial Analysis", "Spatial Data Processing"], [85, 80, 80, 85, 75]],
       'Project 12: Soccer Anyalsis':[["Python", "Data Analysis", "Data Visualization", "JSON"], [80, 80, 75, 70]]

    }

    details_list = data_dict[project]
    plot_list = chart_dict[project]
    datap = {"Skill": plot_list[0], "Percentage": plot_list[1]}

    if len(details_list) <= 1:
        cold,cole = st.columns(2)

        with cold:
            st.subheader("Description:")
            st.write(details_list[0])
        with cole:
            chart1 = create_horizontal_bar_chart(datap, "Skills Used")
            st.plotly_chart(chart1, config={"displayModeBar": False}, use_container_width=True)

    if len(details_list) > 1:
        cold,cole = st.columns(2)

        with cold:
            st.subheader("Description:")
            st.write(details_list[0])
            chart1 = create_horizontal_bar_chart(datap, "Skills Used")
            st.plotly_chart(chart1, config={"displayModeBar": False}, use_container_width=True)

        with cole:
            st.image(details_list[1])
            st.image(details_list[2])
            st.image(details_list[3])


def display_work_experience(year):
    if year <= 2019:
        st.subheader("Assistant Business Manager - PTCL (Jan 2017- Aug 2020)")
        st.write("- Consulting with marketing department to identify high churn, potential and declining areas via SQL, Python and Geo spatial mappings.")
        st.write("- Applied predictive analytics and machine learning to model future sales, churn, and faults statistically.")
    elif year > 2019 and year < 2021:
        st.subheader("Manager Special Projects - PTCL (Sept 2020 - April 2021)")
        st.write("- Interpret patterns and trends to improve business and operational opportunities.")
        st.write("- Develop market analysis for opportunities, analysing and targeting growth for potential clients via machine learning.")
        st.write("- Drive marketing data using data science for actionable data driven marketing analytics")
    elif year > 2020 and year < 2023:
        st.subheader("Data Scientist Intern - PwC Deutschland (Sept 2022 - Feb 2023)")
        st.write("- Developing explainable AI (XAI) code and platform for improving business trust on machine learning based solutions in R & Python.")
        st.write("- Implementing predictive analytics and statistical solutions for forecasting product.")
        st.write("- Evaluating and improving existing machine learning model to improve predicting metrics.")
        st.write("- Optimizing machine learning pipeline using PySpark and parallelization.")
        st.write("- Implementing forecasting solutions in Palantir Foundry platform")
        st.write(" ")
        st.subheader("Data Scientist WerkStudent - Novuter GmbH (Sept 2021 - August 2022)")
        st.write("- Create competitive intelligence dashboard by mining , cleaning, and transforming data from various sources.")
        st.write("- Probabilistic and Bayesian modelling for product trend forecast.")
        st.write("- Data mining and Advance SQL Scripting for analysis.")
        st.write("- Convert data into analytical & actionable insights by predictive analytics and forecasting.")
        st.write("- Python Development and Database Management.")
    else:
        st.subheader("Data Scientist Endress+Hauser(Sept 2023 -present)")
        st.subheader("Data Scientist - Master's Thesis (March 2023 - Aug 2023)")
        st.write("- Developed and implemented Deep Learning models, specifically Autoencoders, to detect fraudulent transactions in SAP")
        st.write("- Conducted thorough Exploratory Data Analysis (EDA) to gain insights and leveraged advanced techniques in anomaly detection and unsupervised learning to improve the accuracy of fraud detection.")
        st.write("- Spearheaded a project focused on fine tuning the ChatGPT model using Azure ML and Azure OpenAI modules to train an internal Large Language Model (LLM) tailored for extracting valuable \
        insights for the company. Also Implemented Azure Cognitive Search and Vector Search Solution to improve LLM based on Retrieval-augmented generation - RAG.")
        st.write("- Developed a user friendly frontend using Streamlit and Fiori app in SAP, enabling seamless access and utilization of the trained LLM by both technical and non technical users.")



def create_skill_chart(skill, proficiency, description, icon):
    fig = go.Figure(go.Pie(
        values=[proficiency, 100 - proficiency],
        labels=[f"({proficiency}%)", ""],
        hole=0.6,
        textinfo='label',
        hoverinfo='none'
    ))
    fig.update_traces(marker=dict(colors=['blue', 'white']))
    #fig.update_layout(title=f"{skill}", title_font={'size': 24}, showlegend=False)
    #experimental
    # Adding a title in the middle of the donut chart
    fig.add_annotation(
        go.layout.Annotation(
            text=f"<b>{skill}</b>",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=24),
            align='center',
            valign='middle'
        )
    )
    fig.update_layout(showlegend=False)
    return fig, description


def main():
    st.set_page_config(page_title="Saad's Resume", page_icon=":memo:", layout="wide")


    # Chatbot
    st.sidebar.write("Toggle Checkbox for a Chat")
    if st.sidebar.checkbox("Ask Me! via Chat"):
        st.sidebar.write("Chatbot [Not a LLM yet]")
        user_name = st.sidebar.text_input("What should I call you?")
        user_company = st.sidebar.text_input("Your Company")

        if user_name and user_company:
            st.sidebar.write(f"Hello, {user_name}! Glad to know you work for {user_company}.")

            questions = [
                "What is your experience in Azure?",
                "What is your experience in Data Science?",
                "What is your experience in Machine Learning?",
                "Have you worked with LLM's?",
                "What can you share about your skills in XAI?",
                "What do you do in your free time?"
            ]

            selected_question = st.sidebar.selectbox("Choose a question", questions)
            if st.sidebar.button("Ask"):
                if selected_question == questions[0]:
                    st.sidebar.write("I have 3+ years of experience in Azure, I've utilized various Azure services for cloud computing, data analytics and AzureML. My expertise includes deploying applications, \
                    and integrating AI services like Azure Cognitive Search, Vector Search to extract insights from structured and unstructured data using LLM.")
                elif selected_question == questions[1]:
                    st.sidebar.write("I have 3+ years of experience in Data Science. Over the course of my career, I've worked on a diverse range of projects that involve data collection, preprocessing, \
                    exploratory analysis, model building, and deployment. \
                    I've successfully applied machine learning algorithms to solve complex problems and have a strong proficiency in tools like Python, SQL, and data visualization libraries. My experience has \
                    also involved collaborating with cross-functional teams and presenting insights derived from data-driven analyses.")
                elif selected_question == questions[2]:
                    st.sidebar.write("I have 3+ year of experience in Machine Learning. Throughout my journey, I've developed a deep understanding of various machine learning algorithms and techniques, \
                    from supervised and unsupervised learning to deep learning. I've successfully implemented machine learning models for tasks like classification, regression, clustering, and natural language processing. \
                    My experience includes feature engineering, model evaluation, and hyperparameter tuning to achieve optimal results. ")
                elif selected_question == questions[3]:
                    st.sidebar.write("Absolutely, I've had the opportunity to work extensively with LLMs. In my recent projects, I've not only utilized LLMs like GPT-3.5 \
                    and GPT4 \
                    but I've also delved into more advanced techniques. For instance, I've employed embeddings to create vector representations of text, enabling efficient \
                    semantic and vector-based searches using Azure Cognitive Search.\
                     By combining the power of LLMs with Azure's capabilities, I've been able to develop enterprise-level chatbots that offer intelligent responses, relevant information retrieval, \
                     and contextual conversations, providing users with valuable insights and enhanced experiences.")
                elif selected_question == questions[4]:
                    st.sidebar.write("I  have experience implementing techniques that enhance model transparency and interpretability, ensuring that complex AI models can be understood by both technical and \
                    non-technical stakeholders. I've utilized feature importance analysis, LIME, SHAP values, and other methods to shed light on model decisions. My proficiency in XAI allows me to create trustworthy \
                    and explainable AI solutions that are essential for building confidence and understanding in AI-driven outcomes.")
                elif selected_question == questions[5]:
                    st.sidebar.write(" I am an avid reader, often immersing myself in a variety of subjects. Occasionally, I indulge in writing poetry as well. Additionally, I endeavor to publish at least one article each \
                    month on Medium, sharing insights from my latest research. During weekends, I find solace in hiking or cycling amidst nature's beauty. Moreover, I cherish spending some of my evenings watching the sunsets, \
                    allowing me to unwind and reflect. Apart from this I also provide community support in opensource AI projects")


    # Profile picture and About Me section
    profile_picture = "Saad.png"
    about_me_bg_color = "#f0f0f0"
    about_me_text_color = "#333333"

    about_me_styles = f"""
        <style>
            .about_me {{
                background-color: {about_me_bg_color};
                color: {about_me_text_color};
                padding: 1rem;
                margin-bottom: 1rem;
                border-radius: 10px;
            }}
        </style>
    """
    st.markdown(about_me_styles, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image(profile_picture) #, width=250

    with col2:
        st.markdown('<div class="about_me">', unsafe_allow_html=True)
        st.header("Muhammad Saad Uddin")
        st.subheader("Data Scientist & AI Enchanté")
        st.write("A wholeheartedly passionate data scientist with an unquenchable thirst for AI advancements.  \
         I thrive on unraveling complex challenges and staying at the vanguard of technological advancements. \
         Fueled by an unyielding curiosity, I consistently go the extra mile and persevere unwaveringly, forging ahead in my pursuit of excellence. \
         Beyond the realm of algorithms and numbers, I am also a poet, finding solace in crafting words that touch the heart and stir the imagination. 🌟")
        st.markdown('</div>', unsafe_allow_html=True)

    #st.title("My Resume")

    #st.markdown("---")
    st.markdown("# Personal Information")
    #st.markdown("---")

    col1a, col2a = st.columns([1, 1])

    with col1a:
        # Phone
        st.markdown("![Foo](https://cdn1.iconfinder.com/data/icons/ionicons-fill-vol-2/512/phone-portrait-48.png) Phone is not available as this is a public site")

        # Email
        st.markdown("![Foo](https://cdn2.iconfinder.com/data/icons/custom-ios-14-1/60/Mail-48.png) Email is not available for the same reason")

        # Website
        st.markdown("[![Foo](https://cdn1.iconfinder.com/data/icons/business-startup-14/60/Development-48.png)](https://saadmuhammad.github.io/) But you can visit my webpage by clicking on icon")

        #coursera link
        st.markdown("[![Foo](https://cdn2.iconfinder.com/data/icons/knowledge-promotion-3/64/education_school_teaching_learning_knowledge_training-48.png)](https://www.coursera.org/instructor/saad-muhammad) My Instructor Profile @ Coursera, click on icon")

    with col2a:
        # LinkedIn
        st.markdown("[![Foo](https://cdn2.iconfinder.com/data/icons/social-media-2285/512/1_Linkedin_unofficial_colored_svg-48.png)](https://www.linkedin.com/in/muhammad-saad17/) Click on icon to see my LinkedIn")

        # GitHub
        st.markdown("[![Foo](https://img.icons8.com/material-outlined/48/000000/github.png)](https://github.com/SaadMuhammad) Explore my GitHub by clicking on icon")

        # Medium
        st.markdown("[![Foo](https://cdn1.iconfinder.com/data/icons/social-media-circle-7/512/Circled_Medium_svg5-48.png)](https://medium.com/@itssaad.muhammad) Follow me on Medium for latest Development in AI by clicking on icon")

        #Credily link
        st.markdown("[![Foo](https://cdn2.iconfinder.com/data/icons/boxicons-solid-vol-1/24/bxs-badge-check-48.png)](https://www.credly.com/users/saad-muhammad/badges) Verify my achievements at Credly by clicking on icon")


    st.write(" ")
    st.write(" ")
    # Work experience with interactive timeline
    st.markdown("---")
    st.header("Work Experience")
    #st.markdown("---")
    year = st.slider("Select a year:", min_value=2017, max_value=2023, value=2023, step=1)
    display_work_experience(year)


    st.write(" ") #wrk should be before education
    st.write(" ")
    st.write("---")
    st.header("Education")
    #st.markdown("---")
    st.subheader("MS Artificial Intelligence and Data Science")
    st.write("Technische Hochschule Deggendorf, Germany (2021 - 2023)")
    st.write("Specialization: Data Science, Explainable AI(XAI), Computer Vision,  Large Language Models and Generative Pre Trained Transformers (GPT)")
    st.write(" ")
    st.subheader("Bachelors of Engineering")
    st.write("NED UET, Pakistan (2012 - 2016)")
    st.write("Specialization: Telecommunications and Computer Science")

    st.write(" ")
    st.write(" ")
    st.markdown("---")
    st.header("Projects")
    #st.markdown("---")
    projects = [
        "Select a project",
        "Project 1: Finetuning ChatGPT via Azure OpenAI",
        "Project 2: Integrating ChatGPT with Azure Cognitive Search and Langchain",
        "Project 3: Movie Recommendation System on Big Data",
        "Project 4: Electric Vehicle Mobility detection",
        "Project 5: UnConstrained optimization using PSO",
        "Project 6: Object detection using transfer learning",
        "Project 7: Feature Selection via Boruta",
        "Project 8: Explainable AI (XAI) and Palantir Foundry",
        "Project 9: Predicting Defaulters Based on Previous Payments",
        "Project 10: Human Activity Recognition(HAR)",
        "Project 11: Analyzing Spatial Business Opportunites via Follium",
        "Project 12: Soccer Anyalsis"
    ]

    selected_project = st.selectbox("Click on a project to see more details:", projects)

    if selected_project != "Select a project":
        with st.expander("Click here to expand and learn more about this project roject"):
            display_project_details(selected_project)


    st.write(" ")
    st.write(" ")
    st.markdown("---")
    st.markdown("# Skills")
    st.write("Click on a skill to expand details")
    #st.markdown("---")

    skills = [
        (" Python", 90, "Advance level expertise in Python with 5+ years of experince.", "Nil"),
        (" Data Science", 90, "Experienced data scientist with over 3 years of hands-on expertise in leveraging data to extract insights and drive informed decisions.\
         Proficient in a wide range of data analysis, visualization, and machine learning techniques.", "Nil"),
        (" Machine Learning", 90, "Experienced machine learning practitioner with 3+ years of hands-on expertise, adept at developing and deploying machine learning models to\
         solve complex problems and enhance decision-making processes.", "Nil"),
        ("Deep Learning", 85, "Skilled in designing and implementing neural networks for solving intricate problems and advancing AI capabilities.", "Nil"),
        ("SQL", 90, "Seasoned SQL professional with over 5 years of extensive experience in database management, query optimization, and data manipulation, ensuring efficient data handling and retrieval.", "Nil"),
        ("R", 80, "Proficient R programmer with 2 years of experience, well-versed in statistical analysis, data visualization, and script development, contributing to data-driven insights and decision-making.", "Nil"),
        ("C/C++", 70, "Proficient in software development, algorithm implementation, and system-level programming, contributing to efficient and robust solutions.", "Nil"),
        ("Explainable AI(XAI)", 85, "Extensive proficiency in Explainable AI (XAI), adept at employing advanced techniques to enhance model transparency, interpretability,\
         and trustworthiness, ensuring AI-driven insights are understandable and actionable.", "Nil"),
        ("Fine tuning Large Language Models", 90, "In-depth understanding of optimizing model performance, adapting pre-trained architectures to specific tasks, and achieving state-of-the-art \
        results through meticulous tuning and experimentation.", "Nil"),
        ("Azure Cognitive Search", 85, "Proficient Azure Cognitive Search specialist with hands-on experience in configuring and optimizing search solutions, integrating AI-driven capabilities,\
         and ensuring efficient data discovery and retrieval for enhanced user experiences.", "Nil"),
        ("Model Evaluation for LLM", 80, "Skilled in model evaluation for large language models (LLMs), with a comprehensive understanding of designing robust evaluation metrics, assessing \
        model performance, and refining models through rigorous testing and analysis, contributing to the development of high-quality natural language processing solutions.", "Nil"),
        ("LangChain", 90, "Experienced LangChain practitioner with a deep understanding of leveraging the platform for specialized language tasks, adept at designing and implementing solutions\
         that harness LangChain's capabilities to drive efficient and accurate language processing outcomes.", "Nil"),
        ("Prompt Engineering", 90, "Having  a comprehensive grasp of crafting effective and tailored prompts for language models, skilled in optimizing input phrasing and format to elicit desired\
         responses, enhancing the quality and relevance of AI-generated content.", "Nil"),
        ("PySpark", 85, "Experienced PySpark developer with over 2 years of hands-on expertise, proficient in leveraging PySpark's capabilities for big data processing, ETL operations, and advanced\
         analytics, contributing to efficient and scalable data solutions.", "Nil"),
        ("SparkSQL", 85, "Leveraging SparkSQL for querying and analyzing large datasets, designing efficient data pipelines, and extracting valuable insights through SQL-based operations.", "Nil"),
        ("MLOps", 80, "Skilled in orchestrating end-to-end machine learning workflows, automating deployment processes, and ensuring seamless integration of models into production environments, \
        enhancing the efficiency and reliability of machine learning systems.", "Nil"),
        ("AutoML", 90, "Proficient in AutoML (Automated Machine Learning) with 3+ years of experience, adept at utilizing automated tools and frameworks to streamline model selection, \
        feature engineering, and hyperparameter tuning, enabling efficient and optimized machine learning model creation.", "Nil"),
        ("Financial Forecasting", 75, "Skilled in analyzing historical data, building predictive models, and using statistical techniques to project future financial trends and outcomes, \
        contributing to informed business decisions.", "Nil"),
        ("Azure Synapse", 75, "Proficient in Azure Synapse with hands-on experience, adept at leveraging its capabilities for data integration, analytics, and data warehousing, \
        contributing to efficient data processing, insights generation, and informed decision-making within the Azure ecosystem.", "Nil"),
        ("Probability & Statistics", 85, "Skilled in applying statistical methods, hypothesis testing, and probability theory to analyze data, make informed decisions, and draw meaningful insights.", "Nil"),
        ("Calculus", 85, "Proficient in Calculus, skilled in applying fundamental mathematical principles for analyzing and solving complex problems in fields such as optimization, physics, and engineering.", "Nil"),
        ("Tensorflow", 90, "Experienced in TensorFlow with 3+ years of proficiency, adept at building and deploying deep learning models, leveraging TensorFlow's functionalities for various AI applications,\
         and contributing to the development of advanced machine learning solutions.", "Nil"),
        ("Keras", 90, "Skilled in Keras with over 3 years of experience, proficient in designing, training, and evaluating deep learning models using Keras's user-friendly interface, contributing \
        to the development of efficient and effective neural network solutions.", "Nil"),
        ("OpenAI API", 90, "Proficient in OpenAI API, experienced in leveraging its capabilities to develop applications that generate natural language text, facilitate language understanding, and\
         contribute to creating innovative AI-powered solutions", "Nil"),
        ("Pytorch", 80, "Proficient in PyTorch, skilled in utilizing its deep learning framework for building and training neural networks, implementing advanced machine learning algorithms, and\
         contributing to the development of cutting-edge AI applications.", "Nil"),
        ("Swarm Intelligence", 80, "Harnessing collective behaviors inspired by nature to solve complex problems, applying decentralized approaches to optimization, decision-making, and problem-solving, \
        and contributing to innovative solutions that draw from the principles of self-organization and cooperation", "Nil"),
        ("Microsoft Azure", 90, "Expert in Microsoft Azure, proficient in leveraging its comprehensive cloud services and solutions to architect, deploy, and manage robust and scalable applications,\
         contributing to the development of secure, efficient, and cloud-native technology solutions.", "Nil"),
        ("Data Visualization", 95, "Skilled in creating impactful visual representations of data to convey insights and trends, utilizing various tools and techniques to enhance\
         understanding and facilitate informed decision-making.", "Nil"),
        ("IBM Watson", 75, "Proficient in IBM Watson, experienced in utilizing its AI-powered capabilities to develop applications that leverage natural language processing, computer vision, and data analysis,\
         contributing to the creation of innovative solutions that harness the power of cognitive computing.", "Nil"),
        ("Palantir", 80, "Experienced in Palantir, skilled in harnessing its data integration and analysis capabilities to empower organizations with actionable insights,\
         facilitating effective decision-making and enhancing data-driven operations.", "Nil"),
        ("Power BI", 80, "Proficient in Power BI, skilled in utilizing its robust data visualization and business intelligence tools to transform raw data into actionable insights, \
        creating interactive dashboards and reports that facilitate informed decision-making across organizations.", "Nil"),
        ("Tableau", 80, "Proficient in Tableau, skilled in leveraging its powerful data visualization and analytics platform to create interactive and insightful visualizations, \
        facilitating data-driven decision-making and enabling effective communication of complex information.", "Nil")
    ]

    # Number of buttons per row
    num_buttons_per_row = 8

    # Calculate the number of rows required
    num_rows = -(-len(skills) // num_buttons_per_row)

    # Create buttons and display charts upon clicking
    selected_skill = None
    for i in range(num_rows):
        cols = st.columns(num_buttons_per_row)
        for j in range(num_buttons_per_row):
            idx = i * num_buttons_per_row + j
            if idx < len(skills):
                skill, proficiency, description, icon = skills[idx]
                #button_code = f'<a href="?skill={skill}" style="text-decoration:none;"><button>{skill} <img src="{icon}" width="20" height="20"/></button></a>'
                #cols[j].markdown(button_code, unsafe_allow_html=True
                if cols[j].button(skill):
                #if cols[j].button(st.markdown(icon)):
                #button_content = f'<div style="display:flex; align-items:center;"><img src="{icon}" width="20" height="20" style="margin-right: 10px;">{skill}</div>'
                #if cols[j].button(button_content, key=skill):
                    selected_skill = skill, proficiency, description, icon

    if selected_skill:
        skill, proficiency, description, icon = selected_skill
        fig, desc = create_skill_chart(skill, proficiency, description, icon)

        cola1, cola2 = st.columns(2)

        with cola1:
            st.plotly_chart(fig)

        with cola2:
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.subheader(desc)


    # Certifications section
    st.write(" ")
    st.write(" ")
    st.markdown("---")
    st.write("## Certifications")
    st.write("Click on a certification to expand details")
    #st.markdown("---")
    #certification1 = st.button("Microsoft Certified Azure Data Scientist")
    #if certification1:
        #col1, col2 = st.columns(2)
        #col1.image("/content/test.PNG")
        #data = {"Skill": ["Python", "Data Science"], "Percentage": [80, 60]}
        #chart = create_horizontal_bar_chart(data, "Skills Learned")
        #col2.plotly_chart(chart, config={"displayModeBar": False})

    Certifications = [
                ("Microsoft Azure Certified Data Scientist", "certis/azds.png", ["Python", "Data Science", 'Machine Learning', 'AzureML', "MLOps"], [80, 80, 85, 90, 75]),
                ("Advance Data Science Specialist - IBM", "certis/adv_DS.PNG", ["Python", "Data Science", 'AI', "Apache Spark", "Statistics", "Deep Learning", "Functional programming"], [80, 85, 80, 80, 85, 75, 70]),
                ("IBM Data Science Professional", "certis/DS_Prof.PNG", ["Python", "Data Science", 'AI', "IBM Watson", "Database", "SQL", "Recommendation System"], [80, 80, 75, 85, 80, 80, 75]),
                ("TensorFlow Professional Developer", "certis/tensor.PNG", ["Python", "Computer Vision", "Finetuning", "Explainable AI (XAI)"], [80, 90, 85, 80]),
                ("Machine Learning Engineer with Azure", "certis/udacity.PNG", ["Python", "Data Science", "MLOps", "AI", "AzureML"], [80, 80, 85, 90, 90]),
                ("IBM Applied AI Specialization", "certis/appliedAI.PNG", ["Python", "Data Science", 'AI', "IBM Watson", "API", "Deep Learning", "Chatbots"], [80, 75, 90, 85, 80, 85, 80]),
                ("Maths for Machine Learning Specialization", "certis/ML_ICE.PNG", ["Python", "Data Science", "Probability & Statistics", "Optimization", "Information Theory", "Calculus"], \
                [80, 90, 95, 90, 90, 90]),
                ("Project Management", "certis/HMM.PNG", ["Project Management"], [90])
    ]


    # Number of buttons per row
    num_buttons_per_row1 = 4

    # Calculate the number of rows required
    num_rows1 = -(-len(Certifications) // num_buttons_per_row1)

    # Create buttons and display charts upon clicking
    selected_skill1 = None
    for i in range(num_rows1):
        colss = st.columns(num_buttons_per_row1)
        for j in range(num_buttons_per_row1):
            idx1 = i * num_buttons_per_row1 + j
            if idx1 < len(Certifications):
                certi, link, skil, perc = Certifications[idx1]
                if colss[j].button(certi):
                    selected_skill1 = certi, link, skil, perc

    if selected_skill1:
        certi, link, skil, perc = selected_skill1
        data = {"Skill": skil, "Percentage": perc}
        chart = create_horizontal_bar_chart(data, "Skills Learned")

        cold1, cold2 = st.columns(2)

        with cold1:
            st.image(link)

        with cold2:
            st.plotly_chart(chart, config={"displayModeBar": False})

    # Projects section
    #st.write(" ")
    #st.write(" ")
    #st.markdown("---")
    #st.write("## Projects")
    #st.markdown("---")
    #project1 = st.button("Project 1")
    #if project1:
        #col1C, col2C = st.columns(2)
        #with col1C:
            #st.image("/content/test.PNG", width=300)
            #st.image("/content/test.PNG", width=300)

        #data = {"Skill": ["Python", "Machine Learning"], "Percentage": [70, 80]}
        #chart = create_horizontal_bar_chart(data, "Skills Used")

        #with col2C:
            #st.plotly_chart(chart, config={"displayModeBar": False}, use_container_width=True)


if __name__ == "__main__":
    main()
