import streamlit as st
import openai
import pandas as pd
import streamlit as st
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from streamlit_chat import message



# Set your OpenAI API key

openai.api_key = st.secrets["openai"]["api_key"]


# Tabs in Streamlit
tab1,tab2, tab3, tab4 = st.tabs(['Chatbot For Queries','Biostatistician', 'Programmer', 'Statistical Programmer'])

# File uploader for the user to upload their dataset
uploaded_file = st.sidebar.file_uploader("Upload a file", type=['csv', 'xpt','xlsx'])

if uploaded_file is not None:
    # Read the uploaded dataset based on the file type
    
    if uploaded_file.name.endswith('.xpt'):
        df = pd.read_sas(uploaded_file, format='xport')

    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
  
    else:
        df = pd.read_csv(uploaded_file)
        

    # Convert byte strings to regular strings if necessary
    df = df.map(lambda x: x.decode() if isinstance(x, bytes) else x)

    # Display success message
    st.sidebar.success(f"File {uploaded_file.name} has been successfully uploaded.")

    # Filter numeric columns for selection
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    

# Kaplan-Meier Plot Tab
    with tab2:
        st.subheader('Uploaded Dataset')
        n=st.slider('Choose Top n Samples to show',min_value=0,max_value=100,value=5,key='tab2')
        st.write(df.head(n))
        option = st.selectbox("Select an option", ["Kaplan-Meier Plot", "Generate TFL"])
        if option == "Kaplan-Meier Plot":
            st.subheader('Kaplan-Meier Plot')

            # Dynamic selection of columns
            treatment_col = st.selectbox('Select Treatment Column', numeric_columns, index=0)
            time_col = st.selectbox('Select Time to Event Column', numeric_columns, index=0)
            event_col = st.selectbox('Select Event Occurrence Column', numeric_columns, index=0)

            # Optional selections for stratification, covariates, and annotations
            stratification_col = st.selectbox('Select Stratification Column (optional)', [None] + numeric_columns)
            covariate_col = st.selectbox('Select Covariate Column (optional)', [None] + numeric_columns)
            annotation_col = st.selectbox('Select Annotation Column (optional)', [None] + numeric_columns)

            # Store the state when 'Generate Kaplan-Meier Plot' is clicked
            if st.button("Generate Kaplan-Meier Plot"):
                try:
                    # Check if selected columns are numeric
                    if not pd.api.types.is_numeric_dtype(df[time_col]):
                        st.error(f"The column '{time_col}' must be numeric for the Kaplan-Meier plot.")
                        st.stop()
                    if not pd.api.types.is_numeric_dtype(df[event_col]):
                        st.error(f"The column '{event_col}' must be numeric for the Kaplan-Meier plot.")
                        st.stop()

                    # Prepare the prompt for generating Kaplan-Meier plot code
                    optional_stratification = f"Stratification: '{stratification_col}'" if stratification_col else ""
                    optional_covariate = f"Covariate: '{covariate_col}'" if covariate_col else ""
                    optional_annotation = f"Annotation: '{annotation_col}'" if annotation_col else ""

                    generic_prompt_k = f"""
                    You are a Python programming assistant.
                    - Before processing the data, convert byte strings to regular strings using an appropriate method.
                    - Graph should be attractive and intuitive.
                    - Use plotly if asked to generate plot.
                    Generate Python code to create a Kaplan-Meier plot using the provided dataset.
                    Use the columns: 
                    - Treatment arm: '{treatment_col}' 
                    - Time to event: '{time_col}' 
                    - Event occurrence: '{event_col}' 
                    {optional_stratification}
                    {optional_covariate}
                    {optional_annotation}
                    Ensure the plot is visually appealing, with appropriate labels and a legend.
                    Use the lifelines library for survival analysis and plotly for plotting.
                    The data is already loaded in a DataFrame named 'df'.
                    """

                    # Make an API call to OpenAI to generate Python code based on the generic prompt
                    response = openai.ChatCompletion.create(
                        model="gpt-4o",
                        messages=[{"role": "system", "content": '''You are a Python programming assistant. As results are being displayed in Streamlit, so the final output of each code should be shown using Streamlit functions.
                        **Important**:
                        - Do not use deprecated functions.
                        - Do not include triple quotes or code markers like python.
                        - Return only Python code without any quotation mark or detail.
                        - Use names for columns which are specified.
                        - No explanations, no comments, just the Python code itself, and do not use deprecated functions.'''},
                                {"role": "user", "content": generic_prompt_k}],
                        max_tokens=800,
                        temperature=0.5
                    )

                    # Extract the generated code
                    generated_code = response['choices'][0]['message']['content'].strip()

                    # Store the generated code in session state
                    st.session_state.generated_code = generated_code

                    # Execute the generated code
                    exec(st.session_state.generated_code)


                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")


            # Show Generated Code Button
            if "generated_code" in st.session_state:
                if st.button("Show Generated Code"):
                    st.code(st.session_state.generated_code, language='python')
        # Generate TFL Tab
        elif option == "Generate TFL":
            st.subheader('Generate TFL')
            st.write("Describe what TFL (Table, Figure, Listing) you would like to create based on your data.")

            # Text input for user prompt
            user_prompt = st.text_area("Enter your prompt here:", "e.g., Create a summary table of demographics by treatment.")

            if st.button("Generate TFL"):
                # Prepare the prompt for generating TFL code
                generic_prompt_tfl = f"""
                You are a Python programming assistant.
                Make sure the Tables and listings are downloadable in CSV.
                Generate code to create TFLs based on the following prompt: "{user_prompt}"
                The data is already loaded in a DataFrame named 'df'.
                Ensure the TFLs as specified in ICH E3 guidelines.
                """

                # Make an API call to OpenAI to generate Python code based on the generic TFL prompt
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[{"role": "system", "content": '''You are a Python programming assistant. As results are being displayed in Streamlit, so the final output of each code should be shown using Streamlit functions.
                    **Important**:
                    - Do not use deprecated functions.
                    - Use plotly if asked to generate plot.
                    - Generate standard TFLs as specified in ICH E3 guidelines and should be filterable.
                    - Do not include triple quotes or code markers like python.
                    - Return only Python code without any quotation mark or detail.
                    - Use names for columns which are specified.
                    - No explanations, no comments, just the Python code itself, and do not use deprecated functions.'''},
                            {"role": "user", "content": generic_prompt_tfl}],
                    max_tokens=800,
                    temperature=0.5
                )

                # Extract the generated TFL code
                generated_tfl_code = response['choices'][0]['message']['content'].strip()
                st.session_state.generated_tfl_code = generated_tfl_code

                # Execute the generated TFL code
                exec(generated_tfl_code)
             # Display the generated TFL code using a button
            if "generated_tfl_code" in st.session_state:
                if st.button("Show Generated TFL Code"):
                    st.code(st.session_state.generated_tfl_code, language='python')
                
                


    with tab3:
        st.subheader('Uploaded Dataset')
        n=st.slider('Choose Top n Samples to show',min_value=0,max_value=100,value=5,key='tab3')
        st.write(df.head(n))
        st.subheader('Create TFls or Kaplan-Meier Plot by Just Entring Prompt')
        def generate_code_from_prompt(prompt):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": '''You are a python programming assistant.
                        as results are being displayed in streamlit so the final output of each code should be shown using streamlit functions
                        **Important**:
                        - Do not include triple quotes or code markers like `python`.
                        - Return only Python code without any quotation mark or detail.
                        -make sure the Tables and listings are downloadable csv 
                        - Use names for cols which are specified
                        -Generate standard TFLs as specified in ICH E3 guidelines 
                        -use plotly for plotting if asked
                        - No explanations, no comments, just the Python code itself. and do not use deprecated functions.'''},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0.5
                )
                generated_code = response['choices'][0]['message']['content'].strip()
                return generated_code
            except Exception as e:
                return f"An error occurred: {str(e)}"

        # User prompt
        user_prompt = st.text_area('User Prompt')

        # Generate code and store it in session state
        if st.button("Generate Code"):
            generated_code = generate_code_from_prompt(user_prompt)
            st.session_state.generated_code = generated_code  # Store generated code in session state

        # Display generated code in a text area for editing
        if 'generated_code' in st.session_state:
            code_to_edit = st.text_area('Generated Code', value=st.session_state.generated_code, height=300)

            if st.button("Run Edited Code"):
                try:
                    # Execute the edited code
                    exec(code_to_edit)
                except Exception as e:
                    st.error(f"An error occurred while executing the code: {str(e)}")
        else:
            st.info("Generate code first to see the editable code.")

    with tab4:
            
     
            st.subheader('Uploaded Dataset')
            n = st.slider('Choose Top n Samples to show', min_value=0, max_value=100, value=5, key='tab4')
            st.write(df.head(n))

            st.subheader('Generate TFL (Table, Figure, Listing)')

            # Select numeric and categorical columns
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

            # Custom User Input for the type of TFL
            user_prompt = st.text_area("Describe the type of TFL (e.g., summary table by treatment, comparison plot):")

            # Dynamic selection of columns for customization
            treatment_col = st.selectbox('Select Treatment Column (for grouping):', categorical_columns)
            summary_cols = st.multiselect('Select Columns for Summary:', numeric_columns)
            summary_metrics = st.multiselect('Select Summary Metrics:', ['Mean', 'Median', 'Count', 'Standard Deviation', 'Range'])

            # Grouping options
            additional_group_col = st.multiselect('Select Additional Grouping Columns:', categorical_columns)

            # Plot options
            plot_type = st.selectbox('Select Plot Type:', ['Histogram', 'Bar Plot', 'Pie Chart'])

            # Button to generate the code based on inputs
            if st.button("Generate Code", key='generate_tfl_code'):
                generic_prompt_tfl = f"""
                You are a Python programming assistant.
                Generate standard TFLs as specified in ICH E3 guidelines.
                Generate code to create a TFL (Table, Figure, Listing) based on the following requirements:
                - User description: '{user_prompt}'
                - Treatment arm: '{treatment_col}'
                - Summary columns: '{summary_cols}'
                - Summary metrics: '{summary_metrics}'
                - Additional grouping: '{additional_group_col}'
                - Plot type: '{plot_type}'
                The data is already loaded in a DataFrame named 'df'.
                Include:
                1. A summary table for the selected columns.
                2. A plot for the specified type.
                Ensure the output is suitable for clinical trial reporting and present it using Streamlit.
                """

                # OpenAI API Call to generate Python code
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": '''You are a python programming assistant.as results are being displayed in streamlit so the final output of each code should be shown using streamlit functions
                        **Important**:
                        - Do not include triple quotes or code markers like `python`.
                        - Return only Python code without any quotation mark or detail.
                        -Use names for cols which are specified
                        -Generate standard TFLs as specified in ICH E3 guidelines 
                        -use plotly for plotting if asked
                        - No explanations, no comments, just the Python code itself. and do not use deprecated functions.'''},
                            {"role": "user", "content": generic_prompt_tfl}
                        ],
                        max_tokens=800,
                        temperature=0.5
                    )

                    # Extract generated code from response
                    generated_tfl_code = response['choices'][0]['message']['content'].strip()
                    st.session_state.generated_tfl_code = generated_tfl_code

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

            if 'generated_tfl_code' in st.session_state:
                edited_code = st.text_area("Generated TFL Code (Editable):", value=st.session_state.generated_tfl_code, height=300)

                # Save the edited code to session state when the user modifies it
                if st.button("Save Edited Code"):
                    st.session_state.generated_tfl_code = edited_code
                    st.success("Code saved!")

                # Run the edited code
                if st.button("Run Saved Code"):
                    try:
                        exec(st.session_state.generated_tfl_code)
                    except Exception as e:
                        st.error(f"An error occurred while running the code: {str(e)}")


with tab1:

    # Setting page title and header
    st.markdown("<h1 style='text-align: center;'>Chatbot For Queries</h1>", unsafe_allow_html=True)



    # Initialise session state variables
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
    if 'model_name' not in st.session_state:
        st.session_state['model_name'] = []
    if 'cost' not in st.session_state:
        st.session_state['cost'] = []
    if 'total_tokens' not in st.session_state:
        st.session_state['total_tokens'] = []
    if 'total_cost' not in st.session_state:
        st.session_state['total_cost'] = 0.0

    # Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
    
    model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    counter_placeholder = st.sidebar.empty()
    # counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
    clear_button = st.sidebar.button("Clear Conversation", key="clear")

    # Map model names to OpenAI model IDs
    if model_name == "GPT-3.5":
        model = "gpt-3.5-turbo"
    else:
        model = "gpt-4"

    # reset everything
    if clear_button:
        st.session_state['generated'] = []
        st.session_state['past'] = []
        st.session_state['messages'] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        st.session_state['number_tokens'] = []
        st.session_state['model_name'] = []
        st.session_state['cost'] = []
        st.session_state['total_cost'] = 0.0
        st.session_state['total_tokens'] = []
        # counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")


    # generate a response
    def generate_response(prompt):
        st.session_state['messages'].append({"role": "user", "content": prompt})

        completion = openai.ChatCompletion.create(
            model=model,
            messages=st.session_state['messages']
        )
        response = completion.choices[0].message.content
        st.session_state['messages'].append({"role": "assistant", "content": response})

        # print(st.session_state['messages'])
        total_tokens = completion.usage.total_tokens
        prompt_tokens = completion.usage.prompt_tokens
        completion_tokens = completion.usage.completion_tokens
        return response, total_tokens, prompt_tokens, completion_tokens


    # container for chat history
    response_container = st.container()
    # container for text box
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("You:", key='input', height=100)
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output, total_tokens, prompt_tokens, completion_tokens = generate_response(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            st.session_state['model_name'].append(model_name)
            st.session_state['total_tokens'].append(total_tokens)

            
            if model_name == "GPT-3.5":
                cost = total_tokens * 0.002 / 1000
            else:
                cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

            st.session_state['cost'].append(cost)
            st.session_state['total_cost'] += cost

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
                # st.write(
                    # f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")
                # counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
            
