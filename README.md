Acest proiect implementeaza un chatbot Ai care recomanda carti dupa preferintele utilizatorului.
Recomandarile sunt generate de un sistem RAG cu ChromaDB si embeddings de la OpenAi. Dupa recomandare, chatbotul apeleaza un tool local pentru a furniza rezumatul complet al cartii.

Functionalitati:
- Baza de date de rezumate(book_summaries.md) cu 13 titluri, fiecare avand 3-5 fraze + teme principale.
- Vector Store: ChomaDB persistent, alimentat cu embeddings text-embedding-3-small de la OpenAI.
- Retrieval Semantic: cautare dupa teme sau cuvinte-cheie
- Chatbot CLI: integrat cu GPT-4o-mini

- Tool Calling: functia get_summary_by_title(title) returneaza rezumatul complet pentru cartea recomandata.
