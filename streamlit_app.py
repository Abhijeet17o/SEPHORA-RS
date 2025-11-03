import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Page config
st.set_page_config(
    page_title="🛍️ Product Recommendations",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .product-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONTENT-BASED RECOMMENDER
# ============================================================================
class ContentBasedRecommender:
    def __init__(self, products_df):
        self.products_df = products_df.copy()
        self._prepare_features()
        self._compute_similarity()
    
    def _prepare_features(self):
        """Prepare product features for similarity computation"""
        # Combine text features
        self.products_df['combined_features'] = (
            self.products_df['product_name'].fillna('') + ' ' +
            self.products_df['brand_name'].fillna('') + ' ' +
            self.products_df.get('primary_category', '').fillna('') + ' ' +
            self.products_df.get('secondary_category', '').fillna('') + ' ' +
            self.products_df.get('tertiary_category', '').fillna('')
        )
    
    def _compute_similarity(self):
        """Compute cosine similarity matrix"""
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(self.products_df['combined_features'])
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    def get_recommendations(self, product_id, n=10):
        """Get top N similar products"""
        try:
            idx = self.products_df[self.products_df['product_id'] == product_id].index[0]
        except IndexError:
            return pd.DataFrame()
        
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+1]  # Exclude the product itself
        
        product_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]
        
        recommendations = self.products_df.iloc[product_indices].copy()
        recommendations['similarity'] = similarity_scores
        
        return recommendations

# ============================================================================
# COLLABORATIVE FILTERING RECOMMENDER
# ============================================================================
class CollaborativeFilteringRecommender:
    def __init__(self, reviews_df, products_df):
        self.reviews_df = reviews_df.copy()
        self.products_df = products_df.copy()
        self._build_user_item_matrix()
    
    def _build_user_item_matrix(self):
        """Build user-item rating matrix"""
        self.user_item_matrix = self.reviews_df.pivot_table(
            index='author_id',
            columns='product_id',
            values='rating',
            fill_value=0
        )
        
        # Convert to sparse matrix for efficiency
        self.sparse_user_item = csr_matrix(self.user_item_matrix.values)
        
        # Fit KNN model
        self.model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
        self.model.fit(self.sparse_user_item)
    
    def get_recommendations(self, user_id, n=10):
        """Get top N recommendations for a user"""
        try:
            user_idx = self.user_item_matrix.index.get_loc(user_id)
        except KeyError:
            return pd.DataFrame()
        
        # Find similar users
        distances, indices = self.model.kneighbors(
            self.user_item_matrix.iloc[user_idx].values.reshape(1, -1),
            n_neighbors=11
        )
        
        # Get products rated by similar users
        similar_users = indices.flatten()[1:]  # Exclude the user itself
        
        # Aggregate ratings from similar users
        similar_users_ratings = self.user_item_matrix.iloc[similar_users]
        avg_ratings = similar_users_ratings.mean(axis=0)
        
        # Get products the user hasn't rated
        user_rated = self.user_item_matrix.iloc[user_idx]
        recommendations_scores = avg_ratings[user_rated == 0]
        
        # Sort and get top N
        top_products = recommendations_scores.sort_values(ascending=False).head(n)
        
        # Get product details
        recommended_product_ids = top_products.index.tolist()
        recommendations = self.products_df[
            self.products_df['product_id'].isin(recommended_product_ids)
        ].copy()
        
        # Add predicted ratings
        recommendations['predicted_rating'] = recommendations['product_id'].map(top_products)
        recommendations = recommendations.sort_values('predicted_rating', ascending=False)
        
        return recommendations

# ============================================================================
# HYBRID RECOMMENDER
# ============================================================================
class HybridRecommender:
    def __init__(self, content_rec, collab_rec, content_weight=0.5):
        self.content_rec = content_rec
        self.collab_rec = collab_rec
        self.content_weight = content_weight
        self.collab_weight = 1 - content_weight
    
    def get_recommendations(self, product_id=None, user_id=None, n=10):
        """Get hybrid recommendations"""
        if product_id and user_id:
            # Both available - use hybrid
            content_recs = self.content_rec.get_recommendations(product_id, n=20)
            collab_recs = self.collab_rec.get_recommendations(user_id, n=20)
            
            # Normalize scores
            if 'similarity' in content_recs.columns:
                content_recs['norm_score'] = (
                    content_recs['similarity'] / content_recs['similarity'].max()
                )
            
            if 'predicted_rating' in collab_recs.columns:
                collab_recs['norm_score'] = (
                    collab_recs['predicted_rating'] / collab_recs['predicted_rating'].max()
                )
            
            # Merge and combine scores
            merged = pd.merge(
                content_recs[['product_id', 'norm_score']],
                collab_recs[['product_id', 'norm_score']],
                on='product_id',
                suffixes=('_content', '_collab'),
                how='outer'
            ).fillna(0)
            
            merged['hybrid_score'] = (
                merged['norm_score_content'] * self.content_weight +
                merged['norm_score_collab'] * self.collab_weight
            )
            
            # Get top N
            top_products = merged.nlargest(n, 'hybrid_score')
            
            # Get product details
            recommendations = self.content_rec.products_df[
                self.content_rec.products_df['product_id'].isin(top_products['product_id'])
            ].copy()
            
            recommendations = recommendations.merge(
                top_products[['product_id', 'hybrid_score']],
                on='product_id'
            )
            recommendations = recommendations.sort_values('hybrid_score', ascending=False)
            
            return recommendations
        
        elif product_id:
            return self.content_rec.get_recommendations(product_id, n=n)
        
        elif user_id:
            return self.collab_rec.get_recommendations(user_id, n=n)
        
        return pd.DataFrame()

# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data
def load_data():
    """Load product and review data from data/ folder"""
    try:
        products_df = pd.read_csv('data/products.csv')
        reviews_df = pd.read_csv('data/reviews.csv')
        return products_df, reviews_df
    except FileNotFoundError as e:
        st.error(f"❌ Data files not found: {e}")
        st.info("Make sure you have 'data/products.csv' and 'data/reviews.csv'")
        st.stop()

@st.cache_resource
def get_recommenders(_products_df, _reviews_df):
    """Initialize recommendation systems"""
    with st.spinner("🔧 Building recommendation engines..."):
        content = ContentBasedRecommender(_products_df)
        collaborative = CollaborativeFilteringRecommender(_reviews_df, _products_df)
        hybrid = HybridRecommender(content, collaborative, content_weight=0.6)
    return content, collaborative, hybrid

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Header
    st.markdown('<h1 class="main-header">🛍️ AI Product Recommendation System</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    products_df, reviews_df = load_data()
    content_rec, collab_rec, hybrid_rec = get_recommenders(products_df, reviews_df)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Choose recommendation type
    rec_type = st.sidebar.selectbox(
        "🎯 Recommendation Method",
        ["Content-Based", "Collaborative Filtering", "Hybrid"]
    )
    
    # Number of recommendations
    num_recs = st.sidebar.slider("📊 Number of Recommendations", 5, 20, 10)
    
    st.sidebar.markdown("---")
    
    # Input selection
    input_type = st.sidebar.radio(
        "📥 Input Type", 
        ["Product ID", "User ID", "Search Product"]
    )
    
    product_id = None
    user_id = None
    
    if input_type == "Product ID":
        product_id = st.sidebar.text_input("Enter Product ID", "P412407")
        
        # Show product info if exists
        if product_id:
            product_info = products_df[products_df['product_id'] == product_id]
            if len(product_info) > 0:
                st.sidebar.success(f"✅ Product found: {product_info.iloc[0]['product_name']}")
            else:
                st.sidebar.warning("⚠️ Product ID not found")
    
    elif input_type == "User ID":
        if rec_type == "Content-Based":
            st.sidebar.warning("⚠️ Content-Based needs a Product ID, not User ID")
        else:
            user_id = st.sidebar.text_input("Enter User ID", "1696370280")
            
            # Show user info if exists
            if user_id:
                user_reviews = reviews_df[reviews_df['author_id'] == user_id]
                if len(user_reviews) > 0:
                    st.sidebar.success(f"✅ User found with {len(user_reviews)} reviews")
                else:
                    st.sidebar.warning("⚠️ User ID not found")
    
    else:  # Search Product
        search = st.sidebar.text_input("🔍 Search Product Name")
        if search:
            matches = products_df[
                products_df['product_name'].str.contains(search, case=False, na=False)
            ]
            if len(matches) > 0:
                options = {
                    f"{row['product_name']} - {row['brand_name']}": row['product_id'] 
                    for _, row in matches.head(20).iterrows()
                }
                selected = st.sidebar.selectbox("Select Product", list(options.keys()))
                product_id = options[selected]
            else:
                st.sidebar.error("❌ No products found")
    
    # Stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Statistics")
    st.sidebar.metric("Total Products", f"{len(products_df):,}")
    st.sidebar.metric("Total Reviews", f"{len(reviews_df):,}")
    st.sidebar.metric("Unique Users", f"{reviews_df['author_id'].nunique():,}")
    
    # Generate button
    if st.sidebar.button("🚀 Generate Recommendations", type="primary", use_container_width=True):
        
        recommendations = None
        
        try:
            if rec_type == "Content-Based" and product_id:
                with st.spinner("🎨 Generating content-based recommendations..."):
                    recommendations = content_rec.get_recommendations(product_id, n=num_recs)
            
            elif rec_type == "Collaborative Filtering" and user_id:
                with st.spinner("👥 Generating collaborative recommendations..."):
                    recommendations = collab_rec.get_recommendations(user_id, n=num_recs)
            
            elif rec_type == "Hybrid":
                with st.spinner("🔮 Generating hybrid recommendations..."):
                    recommendations = hybrid_rec.get_recommendations(
                        product_id=product_id if product_id else None,
                        user_id=user_id if user_id else None,
                        n=num_recs
                    )
            
            if recommendations is not None and len(recommendations) > 0:
                st.session_state.recommendations = recommendations
                st.session_state.rec_type = rec_type
                st.session_state.input_product_id = product_id
                st.success(f"✅ Generated {len(recommendations)} recommendations!")
            else:
                st.error("❌ No recommendations found. Please check your input.")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
    
    # Display recommendations
    if 'recommendations' in st.session_state and st.session_state.recommendations is not None:
        recs = st.session_state.recommendations
        
        # Show selected product if content-based
        if st.session_state.rec_type == "Content-Based" and st.session_state.get('input_product_id'):
            selected = products_df[products_df['product_id'] == st.session_state.input_product_id]
            if len(selected) > 0:
                st.markdown("## 🎯 Selected Product")
                col1, col2, col3, col4 = st.columns(4)
                
                product = selected.iloc[0]
                with col1:
                    st.metric("Product", product['product_name'][:30] + "...")
                with col2:
                    st.metric("Brand", product['brand_name'])
                with col3:
                    price = product.get('price_usd', product.get('price', 0))
                    st.metric("Price", f"${price:.2f}")
                with col4:
                    if 'rating' in product and pd.notna(product['rating']):
                        st.metric("Rating", f"⭐ {product['rating']:.2f}")
                
                st.markdown("---")
        
        st.markdown(f"## 🎯 {st.session_state.rec_type} Recommendations")
        
        # Tabs
        tab1, tab2 = st.tabs(["📋 Products", "📊 Analytics"])
        
        with tab1:
            # Display each recommendation
            for idx, (_, row) in enumerate(recs.iterrows(), 1):
                with st.container():
                    st.markdown('<div class="product-card">', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"### #{idx} - {row['product_name']}")
                        st.markdown(f"**Brand:** {row['brand_name']}")
                        
                        if 'primary_category' in row and pd.notna(row['primary_category']):
                            st.caption(f"Category: {row['primary_category']}")
                        
                        # Show score
                        if 'similarity' in row:
                            st.progress(float(row['similarity']))
                            st.caption(f"Similarity Score: {row['similarity']:.3f}")
                        elif 'predicted_rating' in row:
                            st.progress(float(row['predicted_rating']) / 5.0)
                            st.caption(f"Predicted Rating: {row['predicted_rating']:.2f} / 5.0")
                        elif 'hybrid_score' in row:
                            st.progress(float(row['hybrid_score']))
                            st.caption(f"Hybrid Score: {row['hybrid_score']:.3f}")
                    
                    with col2:
                        price_col = 'price_usd' if 'price_usd' in row else 'price'
                        if price_col in row and pd.notna(row[price_col]):
                            st.metric("Price", f"${row[price_col]:.2f}")
                        
                        if 'rating' in row and pd.notna(row['rating']):
                            st.metric("Rating", f"⭐ {row['rating']:.2f}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Score chart
                score_col = None
                score_title = ""
                
                if 'similarity' in recs.columns:
                    score_col = 'similarity'
                    score_title = "Similarity Score"
                elif 'predicted_rating' in recs.columns:
                    score_col = 'predicted_rating'
                    score_title = "Predicted Rating"
                elif 'hybrid_score' in recs.columns:
                    score_col = 'hybrid_score'
                    score_title = "Hybrid Score"
                
                if score_col:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=recs['product_name'].head(10),
                            y=recs[score_col].head(10),
                            marker_color='#667eea',
                            text=recs[score_col].head(10).round(3),
                            textposition='auto',
                        )
                    ])
                    fig.update_layout(
                        title=f"Top 10 {score_title}",
                        xaxis_title="Product",
                        yaxis_title=score_title,
                        height=400,
                        showlegend=False
                    )
                    fig.update_xaxis(tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Price distribution
                price_col = 'price_usd' if 'price_usd' in recs.columns else 'price'
                if price_col in recs.columns:
                    fig = px.histogram(
                        recs,
                        x=price_col,
                        nbins=10,
                        title="Price Distribution",
                        labels={price_col: "Price (USD)"},
                        color_discrete_sequence=['#667eea']
                    )
                    fig.update_layout(
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Brand pie chart
            if 'brand_name' in recs.columns:
                st.markdown("### Brand Distribution")
                brand_counts = recs['brand_name'].value_counts().head(10)
                fig = px.pie(
                    values=brand_counts.values,
                    names=brand_counts.index,
                    title="Top Brands in Recommendations"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Summary stats
            st.markdown("### 📈 Summary Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                price_col = 'price_usd' if 'price_usd' in recs.columns else 'price'
                if price_col in recs.columns:
                    avg_price = recs[price_col].mean()
                    st.metric("Average Price", f"${avg_price:.2f}")
            
            with col2:
                if 'rating' in recs.columns:
                    avg_rating = recs['rating'].mean()
                    st.metric("Average Rating", f"⭐ {avg_rating:.2f}")
            
            with col3:
                unique_brands = recs['brand_name'].nunique()
                st.metric("Unique Brands", unique_brands)
    
    else:
        # Welcome message
        st.info("👈 Use the sidebar to configure and generate recommendations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🎨 Content-Based
            Find products similar to what you're looking at based on:
            - Product name
            - Brand
            - Category
            - Features
            """)
        
        with col2:
            st.markdown("""
            ### 👥 Collaborative Filtering
            Discover products loved by users similar to you based on:
            - User ratings
            - Purchase history
            - User behavior patterns
            """)
        
        with col3:
            st.markdown("""
            ### 🔮 Hybrid
            Get the best of both worlds:
            - Combines content + collaborative
            - More accurate recommendations
            - Better diversity
            """)
        
        st.markdown("---")
        st.markdown("### 🚀 Quick Start Guide")
        st.markdown("""
        1. **Choose** a recommendation method from the sidebar
        2. **Select** input type (Product ID, User ID, or Search)
        3. **Set** the number of recommendations you want
        4. **Click** "Generate Recommendations" button
        5. **Explore** your personalized recommendations!
        """)

if __name__ == "__main__":
    main()
