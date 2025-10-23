import json
import boto3
from transformers import pipeline

# Initialize AWS services
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('customer-queries')

# Initialize NLP model
sentiment_analyzer = pipeline("sentiment-analysis")

def lambda_handler(event, context):
    try:
        # Parse input
        body = json.loads(event['body'])
        query = body['query']
        customer_id = body['customer_id']
        
        # Analyze sentiment
        sentiment_result = sentiment_analyzer(query)[0]
        
        # Simple intent classification
        query_lower = query.lower()
        if any(word in query_lower for word in ['bill', 'payment', 'charge']):
            intent = "billing"
        elif any(word in query_lower for word in ['error', 'bug', 'issue']):
            intent = "technical"
        else:
            intent = "general"
        
        # Store in DynamoDB
        table.put_item(
            Item={
                'customer_id': customer_id,
                'query': query,
                'intent': intent,
                'sentiment': sentiment_result['label'],
                'confidence': str(sentiment_result['score']),
                'timestamp': context.aws_request_id
            }
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'intent': intent,
                'sentiment': sentiment_result['label'],
                'confidence': sentiment_result['score']
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }