import requests
import openai
from flask import Flask, request, jsonify

app = Flask(__name__)

# Set OpenAI API Key
openai.api_key = 'YOUR_OPENAI_API_KEY'

# Function to get Shopify products
def get_shopify_products():
    url = "https://nuvitababy-com.myshopify.com/admin/api/2023-07/products.json"
    headers = {"X-Shopify-Access-Token": "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"}
    page_info = None
    products = []
    
    while True:
        params = {"limit": 250}  # Get maximum number of products per request
        if page_info:
            params["page_info"] = page_info

        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        products.extend(data.get('products', []))
        
        link_header = response.headers.get("Link")
        if link_header:
            links = link_header.split(", ")
            next_link = [link for link in links if "rel=\"next\"" in link]
            if next_link:
                page_info = next_link[0].split("; ")[0].strip("<>").split("page_info=")[1]
            else:
                break
        else:
            break

    print("Fetched products from Shopify: ", len(products))
    return products

# Fetch Shopify data on server startup
shopify_data = get_shopify_products()

@app.route('/query', methods=['GET'])
def query():
    user_input = request.args.get('input', None)
    print(f"User input: {user_input}")
    
    if user_input is None:
        return jsonify({
            'status': 'error',
            'message': 'No input provided'
        }), 400

    # Create a text from the Shopify data for the GPT-4 model
    text_data = "\n".join([f"Product: {product['title']}\nDescription: {product['body_html']}" for product in shopify_data])

    # Generate a response using GPT-4
    response = openai.Completion.create(
        engine="text-davinci-004",
        prompt=text_data + "\n\n" + user_input,
        temperature=0.5,
        max_tokens=100
    )

    return jsonify({
        'status': 'success',
        'response': response.choices[0].text.strip()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
