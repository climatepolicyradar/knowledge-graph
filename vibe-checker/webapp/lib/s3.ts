import { S3Client } from "@aws-sdk/client-s3";
import { GetParameterCommand, SSMClient } from "@aws-sdk/client-ssm";
import { fromContainerMetadata, fromIni } from "@aws-sdk/credential-providers";

const region = process.env.AWS_REGION || "eu-west-1";

/**
 * Use container credentials when running on ECS, otherwise fall back to a local
 * named profile (defaults to "labs").
 */
function getCredentials() {
  const isEcs = !!process.env.AWS_CONTAINER_CREDENTIALS_RELATIVE_URI;
  return isEcs
    ? fromContainerMetadata()
    : fromIni({ profile: process.env.AWS_PROFILE || "labs" });
}

async function getSsmParameter(name: string): Promise<string> {
  const client = new SSMClient({ region, credentials: getCredentials() });
  try {
    const response = await client.send(
      new GetParameterCommand({ Name: name, WithDecryption: false }),
    );
    return (
      response.Parameter?.Value ??
      (() => {
        throw new Error(`Parameter ${name} has no value`);
      })()
    );
  } catch (error) {
    throw new Error(
      `Failed to retrieve ${name}: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
}

export function createS3Client(): S3Client {
  return new S3Client({ region, credentials: getCredentials() });
}

let cachedBucketName: string | null = null;
let inFlightBucketName: Promise<string> | null = null;

export async function getBucketName(): Promise<string> {
  if (cachedBucketName !== null) {
    return cachedBucketName;
  }
  if (inFlightBucketName !== null) {
    return inFlightBucketName;
  }
  inFlightBucketName = (async () => {
    try {
      cachedBucketName = await getSsmParameter("/vibe-checker/bucket-name");
      return cachedBucketName;
    } catch (error) {
      throw new Error(
        `Failed to get bucket name: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    } finally {
      inFlightBucketName = null;
    }
  })();
  return inFlightBucketName;
}
