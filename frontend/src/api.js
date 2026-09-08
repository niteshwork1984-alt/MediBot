const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

// This error preserves a safe API error message for the visual feedback shown to the user.
export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

// This helper converts an HTTP response into JSON or a safe readable error.
async function readResponse(response) {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new ApiError(body.detail ?? "The request could not be completed.", response.status);
  }
  return body;
}

// This function sends credentials only to MediBot's login endpoint.
export async function login(username, password) {
  const response = await fetch(`${API_BASE_URL}/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  return readResponse(response);
}

// This function sends a question and bearer token without ever sending a role field.
export async function askChat(question, accessToken) {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${accessToken}`,
    },
    body: JSON.stringify({ question }),
  });
  return readResponse(response);
}
