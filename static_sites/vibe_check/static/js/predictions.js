document.addEventListener("DOMContentLoaded", function () {
  const searchInput = document.getElementById("search");
  const passagesContainer = document.getElementById("passages-container");
  const visibleCountSpan = document.getElementById("visible-count");
  const passages = document.querySelectorAll(".passage-card");
  const regionSelect = document.getElementById("region");
  const translatedStatusSelect = document.getElementById("translated-status");
  const corpusSelect = document.getElementById("corpus");
  const totalCount = passages.length;

  function updateVisibleCount() {
    const visiblePassages = document.querySelectorAll(
      ".passage-card:not(.search-hidden):not(.region-hidden):not(.translated-hidden):not(.corpus-hidden)"
    );
    if (visibleCountSpan) {
      visibleCountSpan.textContent = visiblePassages.length;
    }
  }

  function highlightText(text, pattern, caseSensitive = false) {
    try {
      const flags = caseSensitive ? "g" : "gi";
      const regex = new RegExp(pattern, flags);
      return text.replace(
        regex,
        (match) =>
          `<mark class="bg-blue-100 px-1 border-b-2 border-blue-300">${match}</mark>`
      );
    } catch (e) {
      return text;
    }
  }

  function searchPassages(searchTerm) {
    if (!searchTerm) {
      passages.forEach((passage) => {
        passage.classList.remove("search-hidden");
        const passageText = passage.querySelector(".passage-text");
        passageText.innerHTML =
          passageText.getAttribute("data-original-text") ||
          passageText.innerHTML;
      });
      updateVisibleCount();
      return;
    }

    passages.forEach((passage) => {
      const passageTextEl = passage.querySelector(".passage-text");
      // Store original text if not already stored
      if (!passageTextEl.getAttribute("data-original-text")) {
        passageTextEl.setAttribute(
          "data-original-text",
          passageTextEl.innerHTML
        );
      }
      const originalText = passageTextEl.getAttribute("data-original-text");
      let isMatch = false;
      let highlightedText = originalText;

      try {
        if (searchTerm.startsWith("/") && searchTerm.endsWith("/i")) {
          // Case-insensitive regex search
          const pattern = searchTerm.slice(1, -2);
          const regex = new RegExp(pattern, "i");
          isMatch = regex.test(originalText);
          if (isMatch) {
            highlightedText = highlightText(originalText, pattern, false);
          }
        } else if (searchTerm.startsWith("/") && searchTerm.endsWith("/")) {
          // Case-sensitive regex search
          const pattern = searchTerm.slice(1, -1);
          const regex = new RegExp(pattern);
          isMatch = regex.test(originalText);
          if (isMatch) {
            highlightedText = highlightText(originalText, pattern, true);
          }
        } else {
          // Basic case-insensitive search
          const escapedTerm = searchTerm.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
          isMatch = originalText
            .toLowerCase()
            .includes(searchTerm.toLowerCase());
          if (isMatch) {
            highlightedText = highlightText(originalText, escapedTerm, false);
          }
        }
      } catch (e) {
        // If regex is invalid, fall back to basic search
        const escapedTerm = searchTerm.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        isMatch = originalText.toLowerCase().includes(searchTerm.toLowerCase());
        if (isMatch) {
          highlightedText = highlightText(originalText, escapedTerm, false);
        }
      }

      passage.classList.toggle("search-hidden", !isMatch);
      passageTextEl.innerHTML = highlightedText;
    });

    updateVisibleCount();
  }

  // Region filter functionality
  function filterByRegion(selectedRegion) {
    passages.forEach((passage) => {
      const passageRegion = passage.dataset.region;
      const shouldShow = !selectedRegion || passageRegion === selectedRegion;
      passage.classList.toggle("region-hidden", !shouldShow);
    });
    updateVisibleCount();
  }

  // Translated status filter functionality
  function filterByTranslatedStatus(selectedStatus) {
    passages.forEach((passage) => {
      const passageTranslated = passage.dataset.translated;
      const shouldShow =
        !selectedStatus || passageTranslated === selectedStatus;
      passage.classList.toggle("translated-hidden", !shouldShow);
    });
    updateVisibleCount();
  }

  // Corpus filter functionality
  function filterByCorpus(selectedCorpus) {
    passages.forEach((passage) => {
      const passageCorpus = passage.dataset.datasetName;
      const shouldShow = !selectedCorpus || passageCorpus === selectedCorpus;
      passage.classList.toggle("corpus-hidden", !shouldShow);
    });
    updateVisibleCount();
  }

  if (searchInput) {
    searchInput.addEventListener("input", (e) => {
      searchPassages(e.target.value);
    });
  }

  if (regionSelect) {
    regionSelect.addEventListener("change", (e) => {
      filterByRegion(e.target.value);
    });
  }

  if (translatedStatusSelect) {
    translatedStatusSelect.addEventListener("change", (e) => {
      filterByTranslatedStatus(e.target.value);
    });
  }

  if (corpusSelect) {
    corpusSelect.addEventListener("change", (e) => {
      filterByCorpus(e.target.value);
    });
  }

  // Initialize shuffle and sort functionality if elements exist
  const shuffleButton = document.getElementById("shuffle");
  const sortLengthButton = document.getElementById("sortLength");

  if (shuffleButton) {
    shuffleButton.addEventListener("click", () => {
      const passagesArray = Array.from(passages);
      passagesArray.sort(() => Math.random() - 0.5);
      passagesContainer.innerHTML = "";
      passagesArray.forEach((passage) =>
        passagesContainer.appendChild(passage)
      );
    });
  }

  if (sortLengthButton) {
    // Add a data attribute to track sort direction
    sortLengthButton.setAttribute("data-sort-direction", "desc");

    sortLengthButton.addEventListener("click", () => {
      const passagesArray = Array.from(passages);
      const currentDirection = sortLengthButton.getAttribute(
        "data-sort-direction"
      );

      // Sort based on current direction
      passagesArray.sort((a, b) => {
        const lengthDiff =
          parseInt(b.dataset.length) - parseInt(a.dataset.length);
        return currentDirection === "desc" ? lengthDiff : -lengthDiff;
      });

      // Toggle direction for next click
      const newDirection = currentDirection === "desc" ? "asc" : "desc";
      sortLengthButton.setAttribute("data-sort-direction", newDirection);

      // Update button text to indicate current sort
      sortLengthButton.textContent = `Sort by Length (${
        newDirection === "desc" ? "↓" : "↑"
      })`;

      // Clear and repopulate container
      passagesContainer.innerHTML = "";
      passagesArray.forEach((passage) =>
        passagesContainer.appendChild(passage)
      );
    });
  }
});
