"""Assistant documentaire Altéa Services : interface Streamlit.
Lancement, depuis la racine du projet :  streamlit run app.py
Prérequis : base_vectorielle/ créée par les notebooks 1 à 4, et Ollama lancé avec le modèle MODELE_LLM."""

from pathlib import Path

import chromadb
import ollama
import streamlit as st
from sentence_transformers import SentenceTransformer

RACINE = Path(__file__).parent
MODELE_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"   # le même qu'à l'indexation
MODELE_LLM = "llama3.2"
SEUIL = 0.5
REPLI = "Je n'ai pas trouvé cette information dans la documentation d'Altéa."


@st.cache_resource
def charger():
    """Charge le modèle d'embedding et la collection une seule fois."""
    model = SentenceTransformer(MODELE_EMBEDDING)
    client = chromadb.PersistentClient(path=str(RACINE / "base_vectorielle"))
    return model, client.get_collection("altea")


model, collection = charger()


def rechercher(question: str, k: int = 3, service: str | None = None) -> list[dict]:
    vecteur = model.encode([question], normalize_embeddings=True)
    res = collection.query(
        query_embeddings=vecteur.tolist(),
        n_results=k,
        where={"service": service} if service else None,
    )
    return [
        {"texte": doc, "source": meta["source"], "page": meta["page"],
         "service": meta["service"], "score": 1 - dist}
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def construire_prompt(question: str, resultats: list[dict]) -> str:
    extraits = "\n\n".join(
        f"[{i}] (source : {r['source']}, page {r['page']})\n{r['texte']}"
        for i, r in enumerate(resultats, start=1)
    )
    return f"""Tu es l'assistant de documentation interne d'Altéa Services.
Réponds en français, uniquement à partir des extraits ci-dessous.

Règles :
1. Si un extrait contient la réponse, donne-la de façon complète et précise, sans ajouter d'autre commentaire.
2. Pour un délai, un montant ou un nombre, reprends-le tel qu'il est écrit dans l'extrait et précise exactement à quoi il s'applique (qui fait quoi, et à quel moment). Ne rattache jamais un chiffre à une autre étape que celle de l'extrait.
3. Termine par une ligne « Sources : » avec le nom du fichier et la page des extraits utilisés.
4. Si, et seulement si, aucun extrait ne contient la réponse, réponds uniquement : "{REPLI}" et rien d'autre.

EXTRAITS :
{extraits}

QUESTION : {question}

RÉPONSE :"""


def generer(prompt: str) -> str:
    reponse = ollama.chat(
        model=MODELE_LLM,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return reponse["message"]["content"]


def repondre(question: str, k: int, service: str | None):
    resultats = rechercher(question, k=k, service=service)
    if not resultats or resultats[0]["score"] < SEUIL:
        return REPLI, resultats
    reponse = generer(construire_prompt(question, resultats))
    if REPLI in reponse and reponse.strip() != REPLI:   # repli ajouté à tort après une vraie réponse
        reponse = reponse.replace(REPLI, "").strip()
    return reponse, resultats


# ---------- Interface ----------
st.set_page_config(page_title="Assistant Altéa Services", page_icon="📚")
st.title("📚 Assistant documentaire Altéa Services")
st.caption("Posez une question sur la documentation interne : la réponse indique ses sources.")

with st.sidebar:
    st.header("Paramètres")
    choix_service = st.selectbox("Service", ["Tous", "RH", "IT", "General"])
    k = st.slider("Nombre d'extraits (k)", min_value=1, max_value=10, value=3)

question = st.chat_input("Votre question (ex. : Combien de jours de télétravail par semaine ?)")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche et rédaction de la réponse..."):
            try:
                service = None if choix_service == "Tous" else choix_service
                reponse, sources = repondre(question, k, service)
            except Exception as e:
                st.error(f"Erreur : {e}\n\nOllama est-il lancé, avec le modèle « {MODELE_LLM} » installé ?")
                st.stop()

        st.write(reponse)

        if sources:
            with st.expander("Sources consultées"):
                for r in sources:
                    st.markdown(f"**{r['source']}** (page {r['page']}, service {r['service']}) : score {r['score']:.2f}")
                    st.caption(r["texte"][:300].replace("\n", " ") + "...")