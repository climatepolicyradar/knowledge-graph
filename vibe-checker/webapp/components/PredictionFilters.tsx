"use client";

import { useEffect, useRef, useState } from "react";

import MaterialIcon from "@/components/MaterialIcon";
import { Select } from "@base-ui-components/react/select";
import { Slider } from "@base-ui-components/react/slider";
import { FilterState } from "@/types/filters";
import { DEBOUNCE, PUBLICATION_YEARS } from "@/lib/constants";

// Shared styling constant for Select.Item components
const SELECT_ITEM_CLASSES = "relative flex w-full cursor-pointer items-center rounded-sm px-2 py-1.5 text-sm outline-none select-none hover:bg-interactive-hover focus:bg-interactive-hover";

interface PredictionFiltersProps {
  filters: FilterState;
  onFilterChange: (filters: FilterState) => void;
  availableCorpusTypes: string[];
  availableRegions: string[];
  totalFiltered?: number;
  totalUnfiltered?: number;
  searchTerms?: string;
}

export default function PredictionFilters({
  filters,
  onFilterChange,
  availableCorpusTypes,
  availableRegions,
  totalFiltered = 0,
  totalUnfiltered = 0,
  searchTerms,
}: PredictionFiltersProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  // Local state for immediate UI feedback during debounced updates
  const [localFilters, setLocalFilters] = useState<FilterState>(filters);
  const debounceTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Update local state when URL-derived filters change
  useEffect(() => {
    setLocalFilters(filters);
  }, [filters]);

  // Debounced update function for smooth interactions
  const debouncedUpdateFilter = (
    key: keyof FilterState,
    value: string | number | boolean | undefined,
  ) => {
    const newValue = value === "" || value === undefined ? undefined : value;
    const newLocalFilters = {
      ...localFilters,
      [key]: newValue,
    };

    // Update local state immediately for smooth UI
    setLocalFilters(newLocalFilters);

    // Clear existing timeout
    if (debounceTimeoutRef.current) {
      clearTimeout(debounceTimeoutRef.current);
    }

    // Set new timeout for URL update
    debounceTimeoutRef.current = setTimeout(() => {
      onFilterChange(newLocalFilters);
    }, DEBOUNCE.FILTERS);
  };

  // Special debounced update for multiple filter keys (like slider range)
  const debouncedUpdateMultipleFilters = (updates: Partial<FilterState>) => {
    const newLocalFilters = {
      ...localFilters,
      ...updates,
    };

    // Update local state immediately for smooth UI
    setLocalFilters(newLocalFilters);

    // Clear existing timeout
    if (debounceTimeoutRef.current) {
      clearTimeout(debounceTimeoutRef.current);
    }

    // Set new timeout for URL update
    debounceTimeoutRef.current = setTimeout(() => {
      onFilterChange(newLocalFilters);
    }, DEBOUNCE.FILTERS);
  };

  // Immediate update for non-text inputs (selects)
  const immediateUpdateFilter = (
    key: keyof FilterState,
    value: string | number | boolean | undefined,
  ) => {
    const newFilters = {
      ...filters,
      [key]: value === "" || value === undefined ? undefined : value,
    };
    onFilterChange(newFilters);
  };

  const clearFilters = () => {
    // Clear any pending debounced updates
    if (debounceTimeoutRef.current) {
      clearTimeout(debounceTimeoutRef.current);
    }
    setLocalFilters({});
    onFilterChange({});
  };

  // Clean up timeout on unmount
  useEffect(() => {
    return () => {
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current);
      }
    };
  }, []);

  const activeFilterCount =
    Object.keys(localFilters).filter(
      (key) => localFilters[key as keyof FilterState] !== undefined,
    ).length + (searchTerms ? 1 : 0); // Include search terms in count

  return (
    <div>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="text-secondary hover:text-primary flex items-center gap-2 text-sm font-medium transition-colors"
          >
            <MaterialIcon
              name={isExpanded ? "expand_more" : "chevron_right"}
              size={16}
              className="transition-transform"
            />
            Filters
          </button>
          {activeFilterCount > 0 && (
            <span className="rounded-full bg-accent-primary px-2 py-1 text-xs font-medium text-white">
              {activeFilterCount}
            </span>
          )}
          {activeFilterCount > 0 && totalUnfiltered > 0 && (
            <div className="text-secondary flex items-center gap-2 text-xs">
              <div className="h-1.5 w-24 overflow-hidden rounded-full bg-bg-tertiary">
                <div
                  className="h-full rounded-full bg-text-primary transition-all duration-300"
                  style={{
                    width: `${(totalFiltered / totalUnfiltered) * 100}%`,
                  }}
                />
              </div>
              <span className="font-mono">
                {totalFiltered} / {totalUnfiltered}
              </span>
            </div>
          )}
        </div>
        {activeFilterCount > 0 && (
          <button
            onClick={clearFilters}
            className="text-secondary hover:text-primary text-sm transition-colors"
          >
            Clear all
          </button>
        )}
      </div>

      {isExpanded && (
        <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2 xl:grid-cols-3">
          {/* Has Predictions Filter */}
          <div>
            <label className="text-secondary mb-2 block text-sm font-medium">
              Show Predictions
            </label>
            <Select.Root
              items={[
                { value: "", label: "All passages" },
                { value: "true", label: "With predictions only" },
                { value: "false", label: "Without predictions only" },
              ]}
              value={
                localFilters.has_predictions === undefined
                  ? ""
                  : localFilters.has_predictions.toString()
              }
              onValueChange={(value) => {
                immediateUpdateFilter(
                  "has_predictions",
                  value === "" ? undefined : value === "true",
                );
              }}
            >
              <Select.Trigger className="input flex w-full items-center justify-between text-sm">
                <Select.Value />
                <Select.Icon>
                  <MaterialIcon
                    name="expand_more"
                    size={16}
                    className="text-secondary"
                  />
                </Select.Icon>
              </Select.Trigger>
              <Select.Portal>
                <Select.Positioner>
                  <Select.Popup className="z-50 min-w-[8rem] overflow-hidden rounded-md border border-border-primary bg-bg-primary shadow-lg">
                    <Select.Item
                      value=""
                      className={SELECT_ITEM_CLASSES}
                    >
                      <Select.ItemText>All passages</Select.ItemText>
                    </Select.Item>
                    <Select.Item
                      value="true"
                      className={SELECT_ITEM_CLASSES}
                    >
                      <Select.ItemText>With predictions only</Select.ItemText>
                    </Select.Item>
                    <Select.Item
                      value="false"
                      className={SELECT_ITEM_CLASSES}
                    >
                      <Select.ItemText>
                        Without predictions only
                      </Select.ItemText>
                    </Select.Item>
                  </Select.Popup>
                </Select.Positioner>
              </Select.Portal>
            </Select.Root>
          </div>
          {/* Translation Status */}
          <div>
            <label className="text-secondary mb-2 block text-sm font-medium">
              Translation Status
            </label>
            <Select.Root
              items={[
                { value: "", label: "All" },
                { value: "true", label: "Translated" },
                { value: "false", label: "Original" },
              ]}
              value={
                localFilters.translated === undefined
                  ? ""
                  : localFilters.translated.toString()
              }
              onValueChange={(value) => {
                immediateUpdateFilter(
                  "translated",
                  value === "" ? undefined : value === "true",
                );
              }}
            >
              <Select.Trigger className="input flex w-full items-center justify-between text-sm">
                <Select.Value />
                <Select.Icon>
                  <MaterialIcon
                    name="expand_more"
                    size={16}
                    className="text-secondary"
                  />
                </Select.Icon>
              </Select.Trigger>
              <Select.Portal>
                <Select.Positioner>
                  <Select.Popup className="z-50 min-w-[8rem] overflow-hidden rounded-md border border-border-primary bg-bg-primary shadow-lg">
                    <Select.Item
                      value=""
                      className={SELECT_ITEM_CLASSES}
                    >
                      <Select.ItemText>All</Select.ItemText>
                    </Select.Item>
                    <Select.Item
                      value="true"
                      className={SELECT_ITEM_CLASSES}
                    >
                      <Select.ItemText>Translated</Select.ItemText>
                    </Select.Item>
                    <Select.Item
                      value="false"
                      className={SELECT_ITEM_CLASSES}
                    >
                      <Select.ItemText>Original</Select.ItemText>
                    </Select.Item>
                  </Select.Popup>
                </Select.Positioner>
              </Select.Portal>
            </Select.Root>
          </div>

          {/* Corpus Type */}
          <div>
            <label className="text-secondary mb-2 block text-sm font-medium">
              Corpus Type
            </label>
            <Select.Root
              items={[
                { value: "", label: "All types" },
                ...availableCorpusTypes.map((type) => ({
                  value: type,
                  label: type,
                })),
              ]}
              value={localFilters.corpus_type || ""}
              onValueChange={(value) =>
                immediateUpdateFilter("corpus_type", value)
              }
            >
              <Select.Trigger className="input flex w-full items-center justify-between text-sm">
                <Select.Value />
                <Select.Icon>
                  <MaterialIcon
                    name="expand_more"
                    size={16}
                    className="text-secondary"
                  />
                </Select.Icon>
              </Select.Trigger>
              <Select.Portal>
                <Select.Positioner>
                  <Select.Popup className="z-50 min-w-[8rem] overflow-hidden rounded-md border border-border-primary bg-bg-primary shadow-lg">
                    <Select.Item
                      value=""
                      className={SELECT_ITEM_CLASSES}
                    >
                      <Select.ItemText>All types</Select.ItemText>
                    </Select.Item>
                    {availableCorpusTypes.map((type) => (
                      <Select.Item
                        key={type}
                        value={type}
                        className={SELECT_ITEM_CLASSES}
                      >
                        <Select.ItemText>{type}</Select.ItemText>
                      </Select.Item>
                    ))}
                  </Select.Popup>
                </Select.Positioner>
              </Select.Portal>
            </Select.Root>
          </div>

          {/* World Bank Region */}
          <div>
            <label className="text-secondary mb-2 block text-sm font-medium">
              Region
            </label>
            <Select.Root
              items={[
                { value: "", label: "All regions" },
                ...availableRegions.map((region) => ({
                  value: region,
                  label: region,
                })),
              ]}
              value={localFilters.world_bank_region || ""}
              onValueChange={(value) =>
                immediateUpdateFilter("world_bank_region", value)
              }
            >
              <Select.Trigger className="input flex w-full items-center justify-between text-sm">
                <Select.Value />
                <Select.Icon>
                  <MaterialIcon
                    name="expand_more"
                    size={16}
                    className="text-secondary"
                  />
                </Select.Icon>
              </Select.Trigger>
              <Select.Portal>
                <Select.Positioner>
                  <Select.Popup className="z-50 min-w-[8rem] overflow-hidden rounded-md border border-border-primary bg-bg-primary shadow-lg">
                    <Select.Item
                      value=""
                      className={SELECT_ITEM_CLASSES}
                    >
                      <Select.ItemText>All regions</Select.ItemText>
                    </Select.Item>
                    {availableRegions.map((region) => (
                      <Select.Item
                        key={region}
                        value={region}
                        className={SELECT_ITEM_CLASSES}
                      >
                        <Select.ItemText>{region}</Select.ItemText>
                      </Select.Item>
                    ))}
                  </Select.Popup>
                </Select.Positioner>
              </Select.Portal>
            </Select.Root>
          </div>

          {/* Publication Year Range */}
          <div>
            <label className="text-secondary mb-2 block text-sm font-medium">
              Publication Year Range
            </label>
            <div className="px-2">
              <Slider.Root
                min={PUBLICATION_YEARS.MIN}
                max={PUBLICATION_YEARS.MAX}
                value={[
                  localFilters.publication_year_start || PUBLICATION_YEARS.MIN,
                  localFilters.publication_year_end || PUBLICATION_YEARS.MAX,
                ]}
                onValueChange={(value: number[]) => {
                  debouncedUpdateMultipleFilters({
                    publication_year_start:
                      value[0] === PUBLICATION_YEARS.MIN ? undefined : value[0],
                    publication_year_end:
                      value[1] === PUBLICATION_YEARS.MAX ? undefined : value[1],
                  });
                }}
                className="relative flex w-full touch-none items-center select-none"
              >
                <Slider.Control className="relative flex h-5 w-full cursor-pointer items-center">
                  <Slider.Track className="relative h-2 w-full grow overflow-hidden rounded-full bg-bg-tertiary">
                    <Slider.Indicator className="absolute h-full bg-text-secondary" />
                  </Slider.Track>
                  <Slider.Thumb
                    index={0}
                    className="block h-5 w-5 rounded-full border-2 border-text-secondary bg-bg-primary shadow transition-colors focus-visible:ring-2 focus-visible:ring-text-secondary focus-visible:ring-offset-2 focus-visible:outline-none disabled:pointer-events-none disabled:opacity-50"
                  />
                  <Slider.Thumb
                    index={1}
                    className="block h-5 w-5 rounded-full border-2 border-text-secondary bg-bg-primary shadow transition-colors focus-visible:ring-2 focus-visible:ring-text-secondary focus-visible:ring-offset-2 focus-visible:outline-none disabled:pointer-events-none disabled:opacity-50"
                  />
                </Slider.Control>
              </Slider.Root>
              <div className="text-primary mt-2 flex justify-between text-xs">
                <span>{localFilters.publication_year_start || PUBLICATION_YEARS.MIN}</span>
                <span>{localFilters.publication_year_end || PUBLICATION_YEARS.MAX}</span>
              </div>
            </div>
          </div>

          {/* Document ID Search */}
          <div>
            <label className="text-secondary mb-2 block text-sm font-medium">
              Document ID
            </label>
            <input
              type="text"
              value={localFilters.document_id || ""}
              onChange={(e) =>
                debouncedUpdateFilter("document_id", e.target.value)
              }
              className="input w-full text-sm"
              placeholder="e.g. CCLW.executive.10310.4923"
            />
          </div>
        </div>
      )}
    </div>
  );
}
