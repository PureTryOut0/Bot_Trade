from twilio.rest import Client

# Substitua com suas credenciais reais do Twilio
TWILIO_ACCOUNT_SID = "ACbac8e876462c1b07dd1f611a9c4c1575"
TWILIO_AUTH_TOKEN = "101c916e9656f4f2e02d74e93a2a256d"

# Sandbox Twilio WhatsApp
TWILIO_WHATSAPP_NUMBER = "whatsapp:+14155238886"
DESTINATARIO_WHATSAPP = "whatsapp:+5511949595295"  # Seu número com DDI

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


