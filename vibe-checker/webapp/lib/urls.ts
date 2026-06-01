/**
 * Helpers for building external URLs (CPR app, Wikibase).
 */

export function buildDocumentUrl(
  documentSlug: string,
  familySlug: string,
  pageNumber: number,
): string {
  const baseUrl =
    process.env.NEXT_PUBLIC_CPR_APP_URL || "https://app.climatepolicyradar.org";
  return `${baseUrl}/documents/${documentSlug}?page=${pageNumber}&id=${familySlug}`;
}

export function buildWikibaseUrl(conceptId: string): string {
  const baseUrl =
    process.env.NEXT_PUBLIC_WIKIBASE_URL ||
    "https://climatepolicyradar.wikibase.cloud";
  return `${baseUrl}/wiki/Item:${conceptId}`;
}
