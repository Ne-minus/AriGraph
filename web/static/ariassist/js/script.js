document.getElementById("send-btn").addEventListener("click", function() {
    const inputField = document.getElementById("chat-input");
    const userMessage = inputField.value.trim();
  
    if (userMessage !== "") {
      addMessage("user", userMessage);
      inputField.value = "";
      
    fetch("/chatbot/", {
        method: "POST",
        headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": getCookie("csrftoken")  // CSRF token for security
        },
        body: JSON.stringify({ message: userMessage })
    })
    .then(response => response.json())
    .then(data => {
        const botMessage = data.bot_message;
        addMessage("bot", botMessage);
    });
    }
  });
  
  function addMessage(sender, message) {
    const chatBox = document.getElementById("chat-box");
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("chat-message");
  
    if (sender === "user") {
      messageDiv.innerHTML = `<div class="message-user">${message}</div>`;
    } else {
      messageDiv.innerHTML = `<div class="message-bot">${message}</div>`;
    }
  
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight; // Auto scroll to bottom
  }
  
  
// Function to get CSRF token for security
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
      const cookies = document.cookie.split(';');
      for (let i = 0; i < cookies.length; i++) {
        const cookie = cookies[i].trim();
        if (cookie.substring(0, name.length + 1) === (name + '=')) {
          cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
          break;
        }
      }
    }
    return cookieValue;
}