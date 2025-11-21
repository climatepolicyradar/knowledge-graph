import { Metadata } from "next";
import { Suspense } from "react";
import PredictionsPageClient from "./PredictionsPageClient";
import LoadingSpinner from "@/components/LoadingSpinner";

type Props = {
  params: Promise<{ concept_id: string; classifier_id: string }>;
};

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { concept_id, classifier_id } = await params;

  return {
    title: `${concept_id}/${classifier_id}`,
  };
}

export default async function PredictionsPage({ params }: Props) {
  const { concept_id, classifier_id } = await params;

  return (
    <Suspense fallback={<LoadingSpinner message="Loading page..." />}>
      <PredictionsPageClient
        conceptId={concept_id}
        classifierId={classifier_id}
      />
    </Suspense>
  );
}