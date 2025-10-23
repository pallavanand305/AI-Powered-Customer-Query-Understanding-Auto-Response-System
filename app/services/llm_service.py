import asyncio
from typing import Dict, List, Optional
import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

class LLMResponseGenerator:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.generator = pipeline("text-generation", model=model_name, tokenizer=model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    async def generate_contextual_response(self, 
                                         query: str, 
                                         intent: str, 
                                         sentiment: str, 
                                         priority: str,
                                         customer_history: Optional[List[Dict]] = None) -> Dict:
        """Generate contextual response using LLM with customer history"""
        
        # Build context from customer history
        context = self._build_context(customer_history, intent, sentiment, priority)
        
        # Create prompt template
        prompt = self._create_prompt_template(query, intent, sentiment, priority, context)
        
        # Generate response using local LLM
        response = await self._generate_with_local_llm(prompt)
        
        # Post-process and validate response
        processed_response = self._post_process_response(response, intent, sentiment)
        
        return {
            "response": processed_response,
            "response_type": self._determine_response_type(intent, priority),
            "suggested_actions": self._suggest_actions(intent, priority),
            "escalation_needed": priority == "HIGH" and sentiment == "NEGATIVE",
            "confidence_score": 0.85  # Placeholder - would calculate actual confidence
        }
    
    def _build_context(self, history: Optional[List[Dict]], intent: str, sentiment: str, priority: str) -> str:
        """Build context from customer interaction history"""
        context_parts = []
        
        if history:
            context_parts.append("Customer History:")
            for interaction in history[-3:]:  # Last 3 interactions
                context_parts.append(f"- {interaction.get('date', 'Unknown')}: {interaction.get('summary', 'No summary')}")
        
        context_parts.extend([
            f"Current Intent: {intent}",
            f"Sentiment: {sentiment}",
            f"Priority: {priority}"
        ])
        
        return "\n".join(context_parts)
    
    def _create_prompt_template(self, query: str, intent: str, sentiment: str, priority: str, context: str) -> str:
        """Create structured prompt for LLM"""
        template = f"""
You are a professional customer service AI assistant. Based on the context and customer query, provide a helpful, empathetic, and actionable response.

Context:
{context}

Customer Query: "{query}"

Guidelines:
- Be empathetic and professional
- Provide specific next steps when possible
- Acknowledge the customer's concern
- Offer solutions or escalation paths
- Keep response concise but complete

Response:"""
        
        return template
    
    async def _generate_with_local_llm(self, prompt: str) -> str:
        """Generate response using local transformer model"""
        try:
            # Use text generation pipeline
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part
            generated_text = response[len(prompt):].strip()
            
            return generated_text if generated_text else "I understand your concern and will help you resolve this issue."
            
        except Exception as e:
            # Fallback response
            return "Thank you for contacting us. I'll make sure your concern is addressed promptly."
    
    def _post_process_response(self, response: str, intent: str, sentiment: str) -> str:
        """Post-process and validate the generated response"""
        # Clean up response
        response = response.strip()
        
        # Ensure response is appropriate length
        if len(response) < 20:
            response = self._get_fallback_response(intent, sentiment)
        elif len(response) > 500:
            response = response[:500] + "..."
        
        # Add intent-specific elements
        if intent == "billing" and "account" not in response.lower():
            response += " Please have your account number ready for faster assistance."
        elif intent == "technical" and "support" not in response.lower():
            response += " Our technical team will investigate this issue."
        
        return response
    
    def _determine_response_type(self, intent: str, priority: str) -> str:
        """Determine the type of response needed"""
        if priority == "HIGH":
            return "immediate_escalation"
        elif intent == "technical":
            return "technical_support"
        elif intent == "billing":
            return "billing_inquiry"
        else:
            return "general_support"
    
    def _suggest_actions(self, intent: str, priority: str) -> List[str]:
        """Suggest follow-up actions based on intent and priority"""
        actions = []
        
        if priority == "HIGH":
            actions.append("Escalate to senior support")
            actions.append("Schedule callback within 1 hour")
        
        if intent == "billing":
            actions.extend([
                "Review account details",
                "Check recent transactions",
                "Verify billing address"
            ])
        elif intent == "technical":
            actions.extend([
                "Run diagnostic tests",
                "Check system status",
                "Provide troubleshooting steps"
            ])
        elif intent == "complaint":
            actions.extend([
                "Document complaint details",
                "Assign to resolution team",
                "Follow up within 24 hours"
            ])
        
        return actions
    
    def _get_fallback_response(self, intent: str, sentiment: str) -> str:
        """Provide fallback responses for different scenarios"""
        fallback_responses = {
            "billing": "I'll help you with your billing inquiry. Let me review your account details.",
            "technical": "I understand you're experiencing technical difficulties. Let me connect you with our technical support team.",
            "complaint": "I sincerely apologize for any inconvenience. Your feedback is important to us, and I'll ensure this matter is resolved.",
            "general": "Thank you for contacting us. I'm here to help you with your inquiry."
        }
        
        base_response = fallback_responses.get(intent, fallback_responses["general"])
        
        if sentiment == "NEGATIVE":
            base_response = "I understand your frustration. " + base_response
        
        return base_response

class ResponsePersonalizer:
    def __init__(self):
        self.customer_profiles = {}
    
    def personalize_response(self, response: str, customer_id: str, customer_data: Dict) -> str:
        """Personalize response based on customer profile"""
        # Add customer name if available
        if customer_data.get("name"):
            response = f"Hello {customer_data['name']}, " + response
        
        # Adjust tone based on customer tier
        if customer_data.get("tier") == "premium":
            response = response.replace("I'll help", "I'll personally ensure")
        
        # Add relevant account information
        if customer_data.get("account_type"):
            response += f" As a {customer_data['account_type']} customer, you have access to priority support."
        
        return response