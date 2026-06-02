import { NextResponse } from "next/server";

export function successResponse<T>(data: T, status = 200) {
  return NextResponse.json({ success: true, data }, { status });
}

export function errorResponse(error: unknown, status = 500) {
  let message = "Unknown error";
  let statusCode = status;

  if (error instanceof Error) {
    message = error.message;

    if (
      error.name === "CredentialsProviderError" ||
      error.message.includes("Token is expired")
    ) {
      message =
        "AWS credentials have expired. Please run 'aws sso login' to refresh your session.";
      statusCode = 503;
    } else if (
      error.message.includes("BUCKET_NAME environment variable is not set")
    ) {
      message =
        "Server configuration error: S3 bucket name is not configured.";
      statusCode = 500;
    } else if (error.message.includes("No body in S3 response")) {
      message = "Failed to retrieve data from storage.";
      statusCode = 502;
    }
  }

  console.error("API Error:", error);
  return NextResponse.json(
    { success: false, error: message },
    { status: statusCode },
  );
}
