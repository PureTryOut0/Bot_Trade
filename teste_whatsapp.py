from twilio.rest import Client

# Substitua com suas credenciais reais do Twilio
TWILIO_ACCOUNT_SID = "xxxxxxxxxxxxxx5"
TWILIO_AUTH_TOKEN = "1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Sandbox Twilio WhatsApp
TWILIO_WHATSAPP_NUMBER = "whatsapp:xxxxxxxxxxxxxxxxxxx"
DESTINATARIO_WHATSAPP = "whatsapp:+xxxxxxxxxxxxxxxxxxxx5"  # Seu número com DDI

def enviar_mensagem_teste():
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        mensagem = client.messages.create(
            body="🚀 Seu Gostoso!",
            from_=TWILIO_WHATSAPP_NUMBER,
            to=DESTINATARIO_WHATSAPP
        )

        print(f"✅ Mensagem enviada com SID: {mensagem.sid}")
    except Exception as e:
        print(f"❌ Erro ao enviar mensagem: {e}")

if __name__ == "__main__":
    enviar_mensagem_teste()


