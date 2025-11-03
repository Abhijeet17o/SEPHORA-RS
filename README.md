# 🛍️ Product Recommendation System - Streamlit Frontend

A beautiful, interactive Streamlit web application for AI-powered product recommendations.

## Features

- ✨ **Three Recommendation Methods**
  - Content-Based Filtering
  - Collaborative Filtering
  - Hybrid Approach

- 🎨 **Beautiful UI**
  - Modern gradient design
  - Interactive product cards
  - Responsive layout

- 📊 **Analytics Dashboard**
  - Recommendation score visualization
  - Price distribution charts
  - Brand and category insights

- 🔍 **Smart Search**
  - Product search by name or brand
  - Filter by category and brand
  - User-friendly product selection

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your data files are in place:
   - `products.csv`
   - `reviews.csv`

3. Update the imports in `streamlit_app.py` to match your recommender classes

## Running the App

```bash
streamlit run streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

## Usage

1. **Select Recommendation Method**: Choose from Content-Based, Collaborative Filtering, or Hybrid
2. **Choose Input Method**: 
   - Select from product list
   - Enter product ID
   - Enter user ID (for collaborative/hybrid)
3. **Configure Settings**: Adjust number of recommendations
4. **Generate**: Click the button to get recommendations
5. **Explore**: View recommendations, analytics, and comparisons

## Customization

### Connecting Your Recommender Systems

Replace the placeholder code in the `initialize_recommenders` function:

```python
@st.cache_resource
def initialize_recommenders(_products_df, _reviews_df):
    from your_module import ContentBasedRecommender, UserCollaborativeFiltering, HybridRecommender
    
    content_based = ContentBasedRecommender(_products_df)
    collaborative = UserCollaborativeFiltering(_reviews_df, _products_df)
    hybrid = HybridRecommender(content_based, collaborative)
    
    return {
        'content': content_based,
        'collaborative': collaborative,
        'hybrid': hybrid
    }
```

### Update Recommendation Logic

In the button click handler, replace the demo code:

```python
if rec_type == "Content-Based" and selected_product_id:
    recs = recommenders['content'].get_recommendations(selected_product_id, n=num_recs)
    st.session_state.recommendations = recs
```

## Architecture

```
streamlit_app.py
├── Data Loading (@st.cache_data)
├── Recommender Initialization (@st.cache_resource)
├── UI Components
│   ├── Sidebar (Settings & Filters)
│   ├── Main Area (Recommendations)
│   └── Tabs (Recommendations, Analytics, Compare)
└── Visualization Functions
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT License
```
