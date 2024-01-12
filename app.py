import numpy as np
import streamlit as st
from tensorflow import keras
from keras.models import load_model
import pickle


def load_models():
    # Load the trained NCF model
    ncf_model = load_model('ncf_model.h5')
    # Load the encoders
    with open('user_encoder.pkl', 'rb') as file:
        user_encoder = pickle.load(file)
    with open('item_encoder.pkl', 'rb') as file:
        item_encoder = pickle.load(file)
    return ncf_model, user_encoder, item_encoder

def get_property_name(id):
    if id == 51:
        return 'Curvo Residences'
    elif id == 50:
        return 'EdgeWood Residences'
    elif id == 46:
        return 'SkyMeridien Residences'
    elif id == 43:
        return 'SkyAwani IV Residence'
    elif id == 47:
        return 'SkyAwani V Residence'
    elif id == 48:
        return 'SkyVogue Residences'
    elif id == 45:
        return 'The Valley Residences'
    elif id == 16:
        return 'Bennington Residences'

def get_property_image(id):
    if id == 51:
        return 'curvo.png'
    elif id == 50:
        return 'edgewood.png'
    elif id == 46:
        return 'skymeridien.png'
    elif id == 43:
        return 'skyawani4.png'
    elif id == 47:
        return 'skyawani5.png'
    elif id == 48:
        return 'skyvogue.png'
    elif id == 45:
        return 'thevalley.png'
    elif id == 16:
        return 'bennington.png'
    
def get_property_desc(id):
    if id == 51:
        return "Natural Living In The Heart Of Setapak. By embracing the beauty of natural elements around us, SkyWorld's latest and iconic Curvo Residences is designed to bring the nature's bounty inside."
    elif id == 50:
        return "EdgeWood Residences offered the seamless integration of lakeside living and rejuvenating tranquility of nature. Green and serene landscapes, a harmonious neighbourhood, EdgeWood offers you a place to truly call your home."
    elif id == 46:
        return "A Luxury City Resort Residence, providing you an ultimate sky living experience in Sentul East.Known for its exceptionally great accessibility to the heart of Kuala Lumpur."
    elif id == 43:
        return "Enjoy carefree days by the park. Introducing a delightful edition of SkyAwani series, the SkyAwani IV Residence, lushly designed with a park frontage that entices limitless living beyong walls!"
    elif id == 47:
        return "A lushly designed condominium with full-fledge facilities that fosters sky high living at its best! SkyAwani 5 Residence marks a notable presence in the uprising vibrant neighborhood of Bandar Baru Sentul where most amenities are within reach."
    elif id == 48:
        return "A Well Crafted Home That Never Goes Out Of Style. A masterpiece of modern minimalism embellished with meticulous attention to detail. A habitat that marries form and function in one captivating persona."
    elif id == 45:
        return 'Located in the heart of Setiawangsa, The Valleys at SkySierra emphasises quality living amidst a nature-inspired environment. It is designed to cater for a convenient, active and vibrant lifestyle featuring 4 different levels of facility deck.'
    elif id == 16:
        return "Discover a lifestyle inspired by wellness where a quietly serene garden welcomes you, an inner city lifestyle that celebrates wellness in harmony with nature."

def get_property_url(id):
    if id == 51:
        return 'https://skyworld.my/curvo/'
    elif id == 50:
        return 'https://skyworld.my/edgewood/'
    elif id == 46:
        return 'https://skyworld.my/skymeridien/'
    elif id == 43:
        return 'https://skyworld.my/skyawani4/'
    elif id == 47:
        return 'https://skyworld.my/skyawani5/'
    elif id == 48:
        return 'https://skyworld.my/skyvogue/'
    elif id == 45:
        return 'https://skyworld.my/skysierra/'
    elif id == 16:
        return 'https://skyworld.my/bennington/'

def generate_containers(decoded_item_ids):
    c = st.container(border=True)
    c.write("Property Recommendation Results:")
    for id in decoded_item_ids:
        c1 = c.container(border=True)
        col1, col2 = c1.columns([1, 3])
        col1.image(get_property_image(id))
        col2.subheader(get_property_name(id))
        col2.write(get_property_desc(id))
        innercol1, innercol2, innercol3 = col2.columns([1, 1, 1])
        urlbutton = innercol2.link_button("Check it out", url=get_property_url(id))

def main():
    st.title('Deep Learning Property Recommendation System')
    st.write('This is a deep learning recommendation model built using Neural Collaborative Filtering framework.')
    
    ncf_model, user_encoder, item_encoder = load_models()
    num_items = 8
    
    with st.form("input_form"):
        # input: encoded_user_id
        encoded_user_id = st.number_input('Enter an encoded user ID (0 - 172758):', placeholder='Encoded User ID', max_value=172758, min_value=0)
        generate_button = st.form_submit_button('Generate recommendations')

    # Create Item ID Array
    all_item_ids = np.array(list(range(num_items)))

    # Generate Predictions
    predictions = ncf_model.predict([np.array([encoded_user_id] * num_items), all_item_ids])

    # Sort and get top k recommendations
    k = 4
    topk_recommendations = all_item_ids[np.argsort(predictions[:, 0])[::-1][:k]]

    # Decode the item_id
    decoded_item_ids = item_encoder.inverse_transform(topk_recommendations)
    decoded_user_id = user_encoder.inverse_transform([encoded_user_id])[0]

    # Display top k recommendations
    if generate_button:
        generate_containers(decoded_item_ids)

if __name__ == "__main__":
    main()
