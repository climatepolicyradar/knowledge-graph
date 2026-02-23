import { Metadata } from "next";
import { Suspense } from "react";
import ConceptPageClient from "./ConceptPageClient";
import LoadingSpinner from "@/components/LoadingSpinner";

type Props = {
  params: Promise<{ concept_id: string }>;
};

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { concept_id } = await params;

  return {
    title: concept_id,
  };
}

export default async function ConceptPage({ params }: Props) {
  const { concept_id } = await params;

  return (
    <Suspense fallback={<LoadingSpinner message="Loading concept page..." />}>
      <ConceptPageClient conceptId={concept_id} />
    </Suspense>
  );
}
