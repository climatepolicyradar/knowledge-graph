import MaterialIcon from "@/components/MaterialIcon";

interface PaginationControlsProps {
  page: number;
  totalFiltered: number;
  pageSize: number;
  hasNextPage: boolean;
  hasPreviousPage: boolean;
  onPageChange: (newPage: number) => void;
}

export default function PaginationControls({
  page,
  totalFiltered,
  pageSize,
  hasNextPage,
  hasPreviousPage,
  onPageChange,
}: PaginationControlsProps) {
  const startIndex = (page - 1) * pageSize + 1;
  const endIndex = Math.min(page * pageSize, totalFiltered);

  return (
    <div className="card mb-6 p-4">
      <div className="flex items-center justify-center gap-4">
        <button
          onClick={() => onPageChange(Math.max(1, page - 1))}
          disabled={!hasPreviousPage}
          className={`inline-flex items-center gap-1 text-sm transition-colors ${
            hasPreviousPage
              ? "text-primary hover:text-accent-primary cursor-pointer"
              : "text-tertiary cursor-not-allowed opacity-50"
          }`}
        >
          <MaterialIcon name="chevron_left" size={20} />
          Previous
        </button>

        <div className="text-sm font-medium text-secondary">
          {totalFiltered > 0
            ? `${startIndex}-${endIndex} of ${totalFiltered}`
            : "0 of 0"
          }
        </div>

        <button
          onClick={() => onPageChange(page + 1)}
          disabled={!hasNextPage}
          className={`inline-flex items-center gap-1 text-sm transition-colors ${
            hasNextPage
              ? "text-primary hover:text-accent-primary cursor-pointer"
              : "text-tertiary cursor-not-allowed opacity-50"
          }`}
        >
          Next
          <MaterialIcon name="chevron_right" size={20} />
        </button>
      </div>
    </div>
  );
}