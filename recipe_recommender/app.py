import pandas as pd
import numpy as np
import streamlit as st
import re
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


# 🔹 Load Dataset
@st.cache_data
def load_data():
    file_path = "C:\\Sharanya\\ML\\Projects\\Projects\\recipe_recommender\\indian_food.csv"  # Update with your actual file path
    df = pd.read_csv(file_path)

    # Replace -1 with mean in prep_time & cook_time
    df["prep_time"] = df["prep_time"].replace(-1, df["prep_time"].mean())
    df["cook_time"] = df["cook_time"].replace(-1, df["cook_time"].mean())
    df["total_time"] = df["prep_time"] + df["cook_time"]
    df.drop(columns=["state", "region"], inplace=True)

    return df


recipes = load_data()

# 🔹 Feature Engineering
# One-Hot Encode 'diet'
one_hot_diet = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_diet = one_hot_diet.fit_transform(recipes[["diet"]])
diet_df = pd.DataFrame(
    encoded_diet, columns=one_hot_diet.get_feature_names_out(["diet"])
)

# One-Hot Encode 'course'
one_hot_course = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_course = one_hot_course.fit_transform(recipes[["course"]])
course_df = pd.DataFrame(
    encoded_course, columns=one_hot_course.get_feature_names_out(["course"])
)


diet_df.index = recipes.index
course_df.index = recipes.index
# TF-IDF on 'ingredients'
vectorizer = TfidfVectorizer()
ingredients_tfidf = vectorizer.fit_transform(recipes["ingredients"].fillna(""))

# Convert TF-IDF to DataFrame
tfidf_df = pd.DataFrame(
    ingredients_tfidf.toarray(), columns=vectorizer.get_feature_names_out()
)
tfidf_df.index = recipes.index

# Combine all features
recipes_encoded = pd.concat([recipes, diet_df, course_df, tfidf_df], axis=1)

# Drop non-numeric columns
recipes_encoded.drop(
    ["name", "course", "ingredients", "diet", "flavor_profile"], axis=1, inplace=True
)
print(recipes_encoded.columns)

# 🔹 Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% variance
recipes_pca = pca.fit_transform(recipes_encoded)

# 🔹 Train KNN Model
knn = NearestNeighbors(n_neighbors=3, metric="euclidean")
knn.fit(recipes_pca)


# 🔹 Recommendation Function
def recommend_recipe(total_time, main_ingredient, veg):
    main_ingredient = main_ingredient.lower()

    # Use correct diet column
    diet_col = "diet_vegetarian" if veg else "diet_non vegetarian"

    # Filter dataset based on user constraints
    filtered_df = recipes_encoded[
        (recipes_encoded["prep_time"] <= total_time)
        & (recipes_encoded[diet_col] == 1)
        & (
            recipes["ingredients"].apply(
                lambda x: bool(re.search(rf"\b{main_ingredient}\b", str(x).lower()))
            )
        )
    ]

    if filtered_df.empty:
        return "No recipes found with these filters."

    # Transform query point using PCA
    else:
        query_point = filtered_df.iloc[0].values.reshape(1, -1)
        query_point_pca = pca.transform(query_point)

        # Find nearest neighbors
        distances, indices = knn.kneighbors(query_point_pca)

        # Return recommended recipes
        return recipes.iloc[indices[0]][
            ["name", "prep_time", "cook_time", "ingredients"]
        ]


# 🔹 Streamlit UI
st.title("🍛 Recipe Recommendation System")
st.write(
    "Find the best recipes based on your available time, main ingredient, and diet preference!"
)

# User Inputs
prep_time = st.slider(
    "⏳ Maximum Prep Time (minutes)",
    min_value=5,
    max_value=200,
    value=45,
    step=5,
)
main_ingredient = st.text_input(
    "🥕 Main Ingredient", placeholder="e.g., Paneer, Potato, Spinach"
)
veg = st.checkbox("🥗 Vegetarian Only", value=True)

# Recommend button
if st.button("🔍 Find Recipes"):
    if main_ingredient.strip() == "":
        st.warning("⚠️ Please enter a main ingredient.")
    else:
        recommendations = recommend_recipe(prep_time, main_ingredient, veg)
        st.write("### 🍽 Recommended Recipes:")
        st.dataframe(recommendations)
