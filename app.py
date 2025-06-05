from flask import Flask, render_template, jsonify, request
import pandas as pd
import zipfile
import re # Import modul re untuk regular expression
import os
import pickle

# --- Tambahan untuk Collaborative Filtering SVD ---
from surprise import SVD, Dataset, Reader
# -------------------------------------------------

# --- Tambahan untuk Similar Movies ---
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# -----------------------------------

# Initialize Flask app
app = Flask(__name__)

# --- PATHS ---
MOVIES_CSV_PATH = './movies.csv'
RATINGS_ZIP_PATH = './ratings_folder.zip'
RATINGS_FOLDER_PATH = './ratings_folder'
RATINGS_EXTRACT_PATH = os.path.join(RATINGS_FOLDER_PATH, 'ratings.csv')
MODEL_DATA_PATH = './model_data.pkl' # PATH UNTUK MODEL YANG DISIMPAN

# --- GLOBAL DATA VARIABLES ---
# DataFrame akan diisi oleh load_or_extract_data()
movies_df = pd.DataFrame()
ratings_df = pd.DataFrame()

# Model dan data terkait model akan diisi oleh initialize_model_and_data()
svd_model = None
trainset = None
all_movie_ids_global = None # Ini akan menjadi list movieId dari movies_df
item_factors_global = None
movie_inner_to_raw_id_map = None
movie_raw_to_inner_id_map = None


# --- Load or Extract Data ---
def load_or_extract_data():
    global movies_df, ratings_df, all_movie_ids_global # Deklarasikan global untuk assignment

    # Extract ratings CSV from ZIP file
    if not os.path.exists(RATINGS_EXTRACT_PATH):
        try:
            if not os.path.exists(RATINGS_FOLDER_PATH):
                os.makedirs(RATINGS_FOLDER_PATH)
                print(f"Created directory: {RATINGS_FOLDER_PATH}")

            with zipfile.ZipFile(RATINGS_ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(RATINGS_FOLDER_PATH)
            print(f"Successfully extracted {RATINGS_ZIP_PATH} to {RATINGS_FOLDER_PATH}")
        except FileNotFoundError:
            print(f"Error: ZIP file {RATINGS_ZIP_PATH} not found.")
            exit()
        except Exception as e:
            print(f"Error extracting ZIP file: {e}")
            exit()
    else:
        print(f"{RATINGS_EXTRACT_PATH} already exists. Skipping extraction.")

    # Load the CSV data
    try:
        current_movies_df = pd.read_csv(MOVIES_CSV_PATH)
        current_ratings_df = pd.read_csv(RATINGS_EXTRACT_PATH)
    except FileNotFoundError as e:
        print(f"Error: CSV file not found: {e}. Make sure movies.csv and ratings.csv (extracted) are present.")
        exit()
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        exit()

    # --- Ekstrak tahun dari judul dan buat kolom 'year' ---
    def extract_year(title):
        match = re.search(r'\((\d{4})\)$', title)
        if match:
            return int(match.group(1))
        return None

    current_movies_df['year'] = current_movies_df['title'].apply(extract_year)
    current_movies_df.dropna(subset=['year'], inplace=True) # Hapus film tanpa tahun yang valid
    current_movies_df['year'] = current_movies_df['year'].astype(int)

    # --- PRE-CALCULATE AVERAGE RATINGS (untuk fitur non-CF) ---
    print("Calculating initial movie ratings summary for non-CF features...")
    movie_ratings_summary_all = current_ratings_df.groupby('movieId').agg(
        mean_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    ).reset_index()
    current_movies_df = pd.merge(current_movies_df, movie_ratings_summary_all, on='movieId', how='left')
    current_movies_df['mean_rating'].fillna(0, inplace=True)
    current_movies_df['num_ratings'].fillna(0, inplace=True)
    current_movies_df['num_ratings'] = current_movies_df['num_ratings'].astype(int)

    # Assign to global DataFrames
    movies_df = current_movies_df
    ratings_df = current_ratings_df
    all_movie_ids_global = list(movies_df['movieId'].unique()) # Inisialisasi all_movie_ids_global di sini

    print("\nMovies Data (after processing and merging ratings):")
    print(movies_df.head(2))
    print(f"Movies DataFrame shape: {movies_df.shape}")
    print("\nRatings Data:")
    print(ratings_df.head(2))
    print(f"Ratings DataFrame shape: {ratings_df.shape}")

# Panggil fungsi untuk memuat data di awal
load_or_extract_data()


# --- SVD COLLABORATIVE FILTERING MODEL & SIMILARITY DATA INITIALIZATION ---
def initialize_model_and_data():
    global svd_model, trainset, all_movie_ids_global, item_factors_global, \
           movie_inner_to_raw_id_map, movie_raw_to_inner_id_map, ratings_df, movies_df # Tambahkan df yang relevan

    if os.path.exists(MODEL_DATA_PATH):
        print(f"Loading model and data from {MODEL_DATA_PATH}...")
        try:
            with open(MODEL_DATA_PATH, 'rb') as f:
                saved_data = pickle.load(f)
            svd_model = saved_data['svd_model']
            trainset = saved_data['trainset']
            # all_movie_ids_global sudah diinisialisasi dari movies_df saat load_or_extract_data
            # Jadi, kita bisa memverifikasi atau membiarkannya. Untuk konsistensi, bisa juga di-load jika ada di pickle.
            # Jika mau load dari pickle, pastikan disimpan juga.
            # Untuk saat ini, kita akan mengandalkan all_movie_ids_global dari movies_df.
            # Jika Anda menyimpan 'all_movie_ids_global' di pickle, Anda bisa uncomment baris berikut:
            # all_movie_ids_global = saved_data.get('all_movie_ids_global', all_movie_ids_global) # Ambil dari pickle jika ada, jika tidak, pertahankan yang sudah ada

            item_factors_global = saved_data['item_factors_global']
            movie_inner_to_raw_id_map = saved_data['movie_inner_to_raw_id_map']
            movie_raw_to_inner_id_map = saved_data['movie_raw_to_inner_id_map']
            print("Model and data loaded successfully.")
            if svd_model and trainset and item_factors_global is not None:
                 print(f"SVD model ready. Trainset has {trainset.n_users} users and {trainset.n_items} items.")
                 print(f"Item factors shape: {item_factors_global.shape if item_factors_global is not None else 'N/A'}")
            else:
                print("Warning: Some components were not loaded correctly from pickle. Retraining might be needed.")
                raise FileNotFoundError
            return
        except Exception as e:
            print(f"Error loading model data from {MODEL_DATA_PATH}: {e}. Retraining model.")
            if os.path.exists(MODEL_DATA_PATH):
                try: os.remove(MODEL_DATA_PATH)
                except OSError as oe: print(f"Error removing file {MODEL_DATA_PATH}: {oe}")


    print("Training new SVD model and preparing data...")
    if ratings_df.empty or not all(col in ratings_df.columns for col in ['userId', 'movieId', 'rating']):
        print("Error: ratings_df is empty or missing required columns for SVD training.")
        return

    reader = Reader(rating_scale=(ratings_df['rating'].min(), ratings_df['rating'].max()))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    
    current_trainset = data.build_full_trainset()
    
    if movies_df.empty: # Pastikan movies_df sudah ada untuk all_movie_ids_global
        print("Error: movies_df not loaded. Cannot determine all_movie_ids_global.")
        return
    # all_movie_ids_global sudah diisi saat load_or_extract_data

    current_svd_model = SVD(n_epochs=25, n_factors=100, biased=True, lr_all=0.007, reg_all=0.05, random_state=42, verbose=True)
    
    print("Fitting SVD model...")
    current_svd_model.fit(current_trainset)
    print("SVD model training complete.")

    print("Preparing data for item similarity calculations...")
    current_item_factors = None
    current_movie_inner_to_raw_id_map = None
    current_movie_raw_to_inner_id_map = None
    try:
        current_item_factors = current_svd_model.qi
        current_movie_inner_to_raw_id_map = {inner_id: current_trainset.to_raw_iid(inner_id) for inner_id in current_trainset.all_items()}
        
        raw_ids_in_trainset = {current_trainset.to_raw_iid(inner_id) for inner_id in current_trainset.all_items()}
        valid_movie_ids_for_map = [mid for mid in movies_df['movieId'].unique() if mid in raw_ids_in_trainset]
        current_movie_raw_to_inner_id_map = {raw_id: current_trainset.to_inner_iid(raw_id) for raw_id in valid_movie_ids_for_map}
        
        print(f"Item similarity data prepared. {len(current_movie_raw_to_inner_id_map)} movies mapped for similarity.")
    except Exception as e:
        print(f"Error preparing item similarity data: {e}")

    svd_model = current_svd_model
    trainset = current_trainset
    item_factors_global = current_item_factors
    movie_inner_to_raw_id_map = current_movie_inner_to_raw_id_map
    movie_raw_to_inner_id_map = current_movie_raw_to_inner_id_map

    if svd_model and trainset and item_factors_global is not None \
       and movie_inner_to_raw_id_map and movie_raw_to_inner_id_map:
        print(f"Saving trained model and data to {MODEL_DATA_PATH}...")
        try:
            with open(MODEL_DATA_PATH, 'wb') as f:
                pickle.dump({
                    'svd_model': svd_model,
                    'trainset': trainset,
                    # 'all_movie_ids_global': all_movie_ids_global, # Opsional untuk disimpan, karena bisa direkonstruksi dari movies_df
                    'item_factors_global': item_factors_global,
                    'movie_inner_to_raw_id_map': movie_inner_to_raw_id_map,
                    'movie_raw_to_inner_id_map': movie_raw_to_inner_id_map
                }, f)
            print("Model and data saved successfully.")
        except Exception as e:
            print(f"Error saving model data: {e}")
    else:
        print("Skipping save, as some essential components were not prepared successfully.")

# Latih model dan siapkan data similarity saat aplikasi dimulai (jika belum ada)
initialize_model_and_data()

# --- Fungsi untuk Similar Movies (menggunakan faktor SVD) ---
def get_similar_movies_svd_factors(target_movie_id_raw, top_n=10):
    global item_factors_global, movie_raw_to_inner_id_map, movie_inner_to_raw_id_map, movies_df # Akses movies_df

    if item_factors_global is None or movie_raw_to_inner_id_map is None or movie_inner_to_raw_id_map is None:
        print("Item similarity components are not initialized.")
        return []
    
    if target_movie_id_raw not in movie_raw_to_inner_id_map:
        print(f"Target movie ID {target_movie_id_raw} not found in raw_to_inner_id_map (not in model's trainset).")
        return []

    try:
        target_movie_inner_id = movie_raw_to_inner_id_map[target_movie_id_raw]
        target_vector = item_factors_global[target_movie_inner_id].reshape(1, -1)

        similarities = cosine_similarity(target_vector, item_factors_global)[0]
        similar_item_indices = np.argsort(similarities)[::-1]

        found_similar_movies = []
        for item_inner_idx in similar_item_indices:
            if item_inner_idx == target_movie_inner_id:
                continue
            if len(found_similar_movies) >= top_n:
                break
            if item_inner_idx < len(item_factors_global) and item_inner_idx in movie_inner_to_raw_id_map:
                raw_id = movie_inner_to_raw_id_map[item_inner_idx]
                movie_detail_df = movies_df[movies_df['movieId'] == raw_id] # Akses movies_df
                if not movie_detail_df.empty:
                    movie_detail = movie_detail_df.iloc[0]
                    found_similar_movies.append({'title': movie_detail['title'], 'movieId': raw_id})
            else:
                pass
        return found_similar_movies
    except IndexError as e:
        print(f"IndexError for target_movie_inner_id: {target_movie_inner_id} (raw: {target_movie_id_raw}). Max index: {len(item_factors_global)-1}. Details: {e}")
        return []
    except KeyError as e:
        print(f"KeyError: Movie ID {e} not found in mapping dictionary.")
        return []
    except Exception as e:
        print(f"Error in get_similar_movies_svd_factors for movie ID {target_movie_id_raw}: {e}")
        return []

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/movies_by_year', methods=['GET'])
def movies_by_year():
    global movies_df # Akses movies_df
    year_str = request.args.get('year', type=str)
    if not year_str or not year_str.strip().isdigit():
        return jsonify({"message": "Please enter a valid numeric year."}), 400
    try:
        year_query = int(year_str.strip())
    except ValueError:
        return jsonify({"message": f"Invalid year format: '{year_str}'. Please enter a numeric year."}), 400

    if movies_df.empty: return jsonify({"message": "Movie data not loaded yet."}), 503
    filtered_movies = movies_df[movies_df['year'] == year_query]
    if filtered_movies.empty:
        return jsonify({"message": f"No movies found for the year {year_query}."})
    top_movies = filtered_movies.sort_values(by=['mean_rating', 'num_ratings'], ascending=[False, False]).head(10)
    if top_movies.empty:
         return jsonify({"message": f"No rated movies found for the year {year_query}."})
    return jsonify([{'title': movie['title'], 'rating': round(movie['mean_rating'], 2)} for _, movie in top_movies.iterrows()])

@app.route('/genre')
def genre_page():
    return render_template('genre.html')

@app.route('/movies_by_genre', methods=['GET'])
def movies_by_genre():
    global movies_df # Akses movies_df
    genre_query = request.args.get('genre', type=str)
    if not genre_query or not genre_query.strip():
        return jsonify({"message": "Please enter a genre."}), 400
    genre_cleaned = genre_query.strip().lower()
    
    if movies_df.empty or 'genres' not in movies_df.columns:
        return jsonify({"message": "Movie data is not available or not loaded correctly."}), 503
        
    filtered_movies = movies_df[movies_df['genres'].str.contains(genre_cleaned, case=False, na=False, regex=False)]
    if filtered_movies.empty:
        return jsonify({"message": f"No movies found for the genre '{genre_query}'."})
    top_movies = filtered_movies.sort_values(by=['mean_rating', 'num_ratings'], ascending=[False, False]).head(10)
    if top_movies.empty:
         return jsonify({"message": f"No rated movies found for the genre '{genre_query}'."})
    return jsonify([{'title': movie['title'], 'rating': round(movie['mean_rating'], 2)} for _, movie in top_movies.iterrows()])

@app.route('/advanced_search')
def advanced_search_page():
    return render_template('advanced_search.html')

@app.route('/advanced_search_results', methods=['GET'])
def advanced_search_results():
    global movies_df # Akses movies_df
    genre = request.args.get('genre', default="", type=str).strip().lower()
    start_year_str = request.args.get('start_year', default="", type=str).strip()
    end_year_str = request.args.get('end_year', default="", type=str).strip()
    min_rating_str = request.args.get('min_rating', default="", type=str).strip()
    
    if movies_df.empty:
        return jsonify({"message": "Movie data is not available or not loaded correctly."}), 503
        
    results = movies_df.copy()
    if genre: results = results[results['genres'].str.contains(genre, case=False, na=False, regex=False)]
    if start_year_str.isdigit(): results = results[results['year'] >= int(start_year_str)]
    if end_year_str.isdigit(): results = results[results['year'] <= int(end_year_str)]
    if min_rating_str:
        try:
            min_r = float(min_rating_str)
            if 0 <= min_r <= 5: results = results[results['mean_rating'] >= min_r]
        except ValueError: pass
    if results.empty: return jsonify([])
    top_movies = results.sort_values(by=['mean_rating', 'num_ratings'], ascending=[False, False]).head(10)
    return jsonify([{'title': m['title'], 'rating': round(m['mean_rating'], 2) if pd.notna(m['mean_rating']) else 'N/A',
                     'year': int(m['year']), 'genres': m['genres'],
                     'num_ratings': int(m['num_ratings']) if pd.notna(m['num_ratings']) else 0}
                    for _, m in top_movies.iterrows()])

@app.route('/recommendations')
def recommendations_page():
    return render_template('recommendations.html')

@app.route('/get_recommendations', methods=['GET'])
def get_svd_recommendations_api():
    global svd_model, trainset, all_movie_ids_global, movies_df # Deklarasi global untuk akses dan potensi modifikasi (all_movie_ids_global)

    user_id_str = request.args.get('user_id', type=str)

    if not svd_model or not trainset:
        print("SVD model or trainset not loaded/trained. Attempting to initialize...")
        initialize_model_and_data() 
        if not svd_model or not trainset:
            return jsonify({"message": "SVD model is not trained or available. Please try again later."}), 503
    
    if not user_id_str or not user_id_str.strip().isdigit():
        return jsonify({"message": "Please enter a valid numeric User ID."}), 400
    
    user_id = int(user_id_str.strip())

    try:
        _ = trainset.to_inner_uid(user_id) 
    except ValueError: 
        if movies_df.empty:
             return jsonify({"message": f"User ID {user_id} not found, and fallback popular movies data is not available."}), 500
        popular_fallback = movies_df.sort_values(by=['mean_rating', 'num_ratings'], ascending=[False, False]).head(10)
        fallback_list = [{'title': movie['title'], 'predicted_rating': round(movie['mean_rating'],2) if pd.notna(movie['mean_rating']) else "N/A"}
                         for _, movie in popular_fallback.iterrows()]
        return jsonify({"message": f"User ID {user_id} not found in training data. Showing popular movies instead.", "recommendations": fallback_list})

    try:
        rated_movie_ids_inner = trainset.ur[trainset.to_inner_uid(user_id)]
        rated_movie_ids = [trainset.to_raw_iid(inner_iid) for inner_iid, _ in rated_movie_ids_inner]
    except ValueError: 
        rated_movie_ids = []
        
    if all_movie_ids_global is None: # Ini seharusnya sudah diisi oleh load_or_extract_data
        if not movies_df.empty:
            # Tidak perlu deklarasi global lagi di sini, karena sudah di atas fungsi
            all_movie_ids_global = list(movies_df['movieId'].unique()) # Assignment ke global
            print("Re-initialized all_movie_ids_global from movies_df in get_recommendations.")
        else:
            return jsonify({"message": "Movie list (all_movie_ids_global) not available for recommendations."}), 500

    unrated_movie_ids = [mid for mid in all_movie_ids_global if mid not in rated_movie_ids]

    if not unrated_movie_ids:
        return jsonify({"message": f"User {user_id} seems to have rated all available movies, or no unrated movies found."})

    predictions = []
    for movie_id in unrated_movie_ids:
        pred = svd_model.predict(uid=user_id, iid=movie_id)
        predictions.append({'movieId': movie_id, 'estimated_rating': pred.est})

    predictions.sort(key=lambda x: x['estimated_rating'], reverse=True)
    top_n_recs = predictions[:10]

    recommended_movies_info = []
    if movies_df.empty:
        return jsonify({"message": "Movie details data is not available to show recommendations."}), 500
        
    for rec in top_n_recs:
        movie_info_df = movies_df[movies_df['movieId'] == rec['movieId']]
        if not movie_info_df.empty:
            recommended_movies_info.append({
                'title': movie_info_df.iloc[0]['title'],
                'predicted_rating': round(rec['estimated_rating'], 2)
            })
    
    if not recommended_movies_info:
        return jsonify({"message": f"Could not generate personalized recommendations for User ID {user_id}. Try another ID."})

    return jsonify(recommended_movies_info)

# --- Routes for Similar Movies Feature ---
@app.route('/similar_movies_page')
def similar_movies_page():
    return render_template('similar_movies.html')

@app.route('/get_similar_movies_by_title', methods=['GET'])
def get_similar_movies_by_title_api():
    global item_factors_global, movie_raw_to_inner_id_map, movies_df # Akses global

    movie_title_query = request.args.get('movie_title')
    if not movie_title_query or not movie_title_query.strip():
        return jsonify({"message": "Please provide a movie title."}), 400
    
    movie_title_query_cleaned = movie_title_query.strip()

    if item_factors_global is None or movie_raw_to_inner_id_map is None:
        print("Item factors or mapping not available. Attempting to initialize model and data...")
        initialize_model_and_data() 
        if item_factors_global is None or movie_raw_to_inner_id_map is None:
            return jsonify({"message": "Model data for similarity calculation is not ready. Please try again later."}), 503

    if movies_df.empty:
        return jsonify({"message": "Movie details data is not available."}), 503

    def clean_title_for_search(title):
        return re.sub(r'\s*\(\d{4}\)$', '', title).lower()

    target_movie_id = None
    target_movie_title_found = None
    cleaned_query = clean_title_for_search(movie_title_query_cleaned)
    
    exact_matches = movies_df[movies_df['title'].apply(clean_title_for_search) == cleaned_query]
    if not exact_matches.empty:
        for _, row in exact_matches.iterrows():
            if row['movieId'] in movie_raw_to_inner_id_map:
                target_movie_id = row['movieId']
                target_movie_title_found = row['title']
                break 
    
    if target_movie_id is None:
        movies_in_model_df = movies_df[movies_df['movieId'].isin(movie_raw_to_inner_id_map.keys())]
        partial_matches = movies_in_model_df[movies_in_model_df['title'].str.lower().str.contains(cleaned_query, regex=False, na=False)]
        
        if not partial_matches.empty:
            partial_matches = partial_matches.copy()
            partial_matches.loc[:, 'title_len_diff'] = partial_matches['title'].apply(lambda t: abs(len(clean_title_for_search(t)) - len(cleaned_query)))
            partial_matches = partial_matches.sort_values(by='title_len_diff')
            
            target_movie_id = partial_matches.iloc[0]['movieId']
            target_movie_title_found = partial_matches.iloc[0]['title']

    if target_movie_id is None:
        any_match_in_db = movies_df[movies_df['title'].str.lower().str.contains(cleaned_query, regex=False, na=False)]
        if not any_match_in_db.empty:
            first_db_match_title = any_match_in_db.iloc[0]['title']
            return jsonify({"message": f"Movie '{first_db_match_title}' found in database, but it may not have enough rating data for similarity. Try another movie.", "searched_title": movie_title_query_cleaned}), 404
        else:
            return jsonify({"message": f"Movie title '{movie_title_query_cleaned}' not found. Please try a different title or be more specific.", "searched_title": movie_title_query_cleaned}), 404

    similar_movies_list = get_similar_movies_svd_factors(target_movie_id, top_n=10)
    
    if not similar_movies_list:
        return jsonify({"message": f"Could not find similar movies for '{target_movie_title_found}'.", "target_movie_title": target_movie_title_found, "searched_title": movie_title_query_cleaned})
            
    return jsonify({"target_movie_title": target_movie_title_found, "similar_movies": similar_movies_list, "searched_title": movie_title_query_cleaned})

# ----------------------------------------------------

if __name__ == '__main__':
    # Memastikan data dan model diinisialisasi jika belum (meskipun sudah dipanggil di atas)
    # Ini sebagai double check sebelum app.run()
    if movies_df.empty:
        print("Main: movies_df is empty, attempting to load data...")
        load_or_extract_data()
    
    if svd_model is None or trainset is None:
        print("Main: SVD model or trainset is None, attempting re-initialization.")
        initialize_model_and_data()

    if not movies_df.empty and svd_model and trainset:
        print("Starting Flask application...")
        app.run(debug=True, port=5001)
    else:
        print("Failed to initialize necessary data or SVD model. Application cannot start.")