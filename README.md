# llms-bootcamp

## **Syllabus :**


### **Emerging Architectures for Large Language Model Applications**


In this module we will understand the common use cases of large language models and fundamental building blocks of such applications. Learners will be introduced to the following topics at a very high level without going into the technical details:


**Large language models and foundation models**
Prompts and prompt engineering
Context window and token limits
Embeddings and vector databases
Build custom LLM applications by:
Training a new model from scratch
Fine-tuning foundation LLMs
In-context learning
Canonical architecture for and end-to-end LLM application


### **Evolution of Embeddings - The Building Blocks of Large Language Models**
In this module, we will be reviewing how embeddings have evolved from the simplest one-hot encoding approach to more recent semantic embeddings approaches. The module will go over the following topics:


Review of classical techniques
Review of binary/one-hot, count-based and TF-IDF techniques for vectorization
Capturing local context with n-grams and challenges
Semantic Encoding Techniques
Overview of Word2Vec and dense word embeddings
Application of Word2Vec in text analytics and NLP tasks
Text Embeddings
Word and sentence embeddings
Text similarity measures
Dot product, Cosine similarity, Inner product

Hands-on Exercise

Creating a TF-IDF embeddings on a document corpus
Calculating similarity between sentences using cosine similarity and dot product


### **Attention Mechanism and Transformers**
Dive into the world of large language models, discovering the potent mix of text embeddings, attention mechanisms, and the game-changing transformer model architecture.

Attention mechanism and transformer models
Encoder decoder
Transformer networks: tokenization, embedding, positional encoding and transformers block
Attention mechanism
Self-Attention
Multi-head Attention
Transformer models
Hands-on Exercise

Understanding attention mechanisms: Self-attention for contextual word analysis


### **Efficient Storage and Retrieval of Vector Embeddings Using Vector Databases**
Learn about efficient vector storage and retrieval with vector database, indexing techniques, retrieval methods, and hands-on exercises.

Overview
Rationale for vector databases
Importance of vector databases in LLMs 
Popular vector databases
Indexing techniques
Product Quantization (PQ), Locality Sensitive Hashing (LSH) and Hierarchical Navigable Small World (HNSW)
Retrieval techniques
Cosine Similarity, Nearest Neighbor Search
Hands-on Exercise
Creating a vector store using HNSW
Creating, storing and retrieving embeddings using cosine similarity and nearest neighbors


### **Leveraging Text Embeddings for Semantic Search**
Understand how semantic search overcomes the fundamental limitation in lexical search i.e. lack of semantic . Learn how to use embeddings and similarity in order to build a semantic search model.

Understanding and Implementing Semantic Search
Introduction and importance of semantic search
Distinguishing Semantic Search from lexical search
Semantic search using text embeddings
Exploring Advanced Concepts and Techniques in Semantic Search
Multilingual search
Limitations of embeddings and similarity in semantic search
Improving semantic search beyond embeddings and similarity
Hands-on Exercise: 
Building a simple semantic search engine with multilingual capability


### **Fundamentals of Prompt Engineering**
Unleash your creativity and efficiency with prompt engineering. Seamlessly prompt models, control outputs, and generate captivating content across various domains and tasks.

Prompt Design and Engineering
Prompting by instruction
Prompting by example
Controlling the Model Output
When to Stop
Being Creative vs. Predictable
Saving and sharing your prompts
Use Case Ideation
Utilizing Goal, Task and Domain for perfect prompt
Example Use Cases
Summarizing (summarizing a technical report)
Inferring (sentiment classification, topic extraction)
Transforming text (translation, spelling and grammar correction)
Expanding (automatically writing emails)
Generating a product pitch
Creating a business model Canvas
Simplifying technical concepts
Composing an email


### **Fine Tuning Foundation LLMs**
In-depth discussion on fine-tuning of large language models through theory discussions, exploring rationale, limitations, and Parameter Efficient Fine Tuning.

Fine Tuning Foundation LLMs
RLHF, Transfer learning and Fine tuning
Limitations for fine tuning
Parameter efficient fine tuning
Quantization of LLMs
Low Rank Adaptation (LoRA) and QLoRA
Fine tuning vs. RAG: When to use one or the other. Risks and limitations.
Hands-on Exercise:
In-Class: Instruction fine-tuning, deploying and evaluating a LLaMA2-7B 4-bit quantized model
Homework: Fine-tuning and deploying OpenAI GPT model on Azure


### **Orchestration Frameworks to Build Applications on Enterprise Data**
Explore the necessity of orchestration frameworks, tackling issues like foundation model retraining, token limits, data source connectivity, and boilerplate code. Discover popular frameworks, their creators, and open source availability.

Why are Orchestration Frameworks Needed?
Eliminate the need for foundation model retraining
Overcoming token limits
Connecters for data sources.


### **LangChain for LLM Application Development**
Build LLM Apps using LangChain. Learn about LangChain's key components such as Models, Prompts, Parsers, Memory, Chains, and Question-Answering. Get hands-on evaluation experience.

Introduction to LangChain
Schema, Models, and Prompts
Memory, Chains
Loading, Transforming, Indexing, and Retrieving data
Document loader
Text splitters
Retrievers
LangChain Use Cases
Summarization – Summarizing long documents
Question & Answering Using Documents As Context
Extraction – Getting structured data from unstructured text
Evaluation – Evaluating outputs generated from LLM models
Querying Tabular Data – without using any extra code
Hands-on Exercise: 
Using LangChain loader, splitter, retrievals on a pdf document


### **Autonomous Agents: Delegating Decision Making to AI**
Use LLMs to make decisions about what to do next. Enable these decisions with tools. In this module, we’ll talk about agents. We’ll learn what they are, how they work, and how to use them within the LangChain library to superpower our LLMs.

Agents and Tools
Agent Types
Conversational agents
OpenAI functions agents
ReAct agents
Plan and execute agents
Hands-on Exercise: Create and execute some of the following agents
Excel agent
JSON agent
Python Pandas agent
Document comparison agent
Power BI agent


### **LLMOps : Observability & Evaluation**
LLMOps encompasses the practices, techniques and tools used for the operational management of large language models in production environments. LLMs offer tremendous business value, humans are involved in all stages of the lifecycle of an LLM from acquisition of data to interpretation of insights. In this module we will learn about the following:

Principles of Responsible AI
Fairness and Eliminating Bias
Reliability and Safety
Privacy and Data Protection
Review techniques for assessing large language model applications, including:
Model fine-tuning
Model inference and serving
Model monitoring with human feedback
Introduce LangKit by WhyLabs for data-centric LLMOps:
Guardrails: Define rules to govern prompts and responses for LLM applications.
Evaluation: Assess LLM performance using known prompts to identify issues.
Observability: Collect telemetry data about the LLM's internal state for monitoring and issue detection.
Hands-on Exercise:
Using Langkit Evaluate LLM performance on specific prompts


### **Evaluating Large Language Models (LLMs)**
Dive into Large Language Model (LLM) evaluation, examining its importance, common issues, and key metrics such as BLEU and ROUGE, and apply these insights through a hands-on summarization exercise.

Introduction to LLM Evaluation

What is evaluation and why is it important for LLMs?
Overview of common mistakes made by LLMs
Brief introduction to benchmark datasets and metrics
Evaluation Metrics

Explain commonly used automatic metrics (BLEU, ROUGE, BERTScore)
Compare strengths and weaknesses of different metrics
Discuss role of human evaluation and techniques (Likert scale)
Hands-on Exercise

Evaluating LLMs summarization using metrics like Rouge and Bertscore


### **Productionize your LLM application**
This module covers how to scale and automate LLM applications using ZenML. ZenML streamlines data versioning, caching, deployment, and collaboration for efficient LLM app development.

Key Challenges in building Enterprise-Level LLM Apps

Introduction about challenges in production
Data Versioning in production
Overview of ZenML's role in scaling LLM apps
Dashboard access for tracking pipeline progress and data storage
Hands-On Exercise

Create a QnA agent with ZenML pipelines
Project: Build A Custom LLM Application On Your Own Data
On the last day of the bootcamp, the learners will apply the concepts and techniques learned during the bootcamp to build an LLM application. Learners will choose to implement one of the following:

Virtual Assistant: A dynamic customer service agent designed for the car manufacturing industry.
Content Generation (Marketing Co-pilot) : Enhancing your marketing strategies with an intelligent co-pilot.
Conversational Agent (Legal and Compliance Assistant): Assisting with legal and compliance matters through interactive conversations.
QnA (IRS Tax Bot): An intelligent bot designed to answer your questions about IRS tax-related topics.
Content Personalizer: Tailoring content specifically to your preferences and needs.
YouTube Virtual Assistant: Engage in interactive conversations with your favorite YouTube channels and playlists.
 

