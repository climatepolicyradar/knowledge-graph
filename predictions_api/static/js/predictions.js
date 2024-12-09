document.addEventListener("DOMContentLoaded", function () {
  const passagesContainer = document.getElementById("passages-container");
  const passages = Array.from(document.querySelectorAll(".passage-card"));
  const visibleCountSpan = document.getElementById("visible-count");
  const searchInput = document.getElementById("search");

  // Populate filter dropdowns
  function populateFilters() {
    const regions = new Set();
    const sources = new Set();
    const types = new Set();

    passages.forEach((passage) => {
      try {
        const metadata = JSON.parse(passage.dataset.metadata);

        if (metadata.world_bank_region) regions.add(metadata.world_bank_region);
        if (metadata.source) sources.add(metadata.source);

        if (metadata.types) {
          try {
            const typeString = metadata.types
              .replace(/^\['|'\]$/g, "")
              .split("', '");
            typeString.forEach((type) => types.add(type));
          } catch (e) {
            console.warn("Could not parse types:", metadata.types);
          }
        }
      } catch (e) {
        console.error("Error parsing metadata:", passage.dataset.metadata);
        console.error(e);
      }
    });
    populateSelect("region", regions);
    populateSelect("source", sources);
    populateSelect("type", types);
  }

  function populateSelect(id, values) {
    const select = document.getElementById(id);
    Array.from(values)
      .sort()
      .forEach((value) => {
        if (value) {
          const option = document.createElement("option");
          option.value = value;
          option.textContent = value;
          select.appendChild(option);
        }
      });
  }

  // Filter functionality
  function applyFilters() {
    const region = document.getElementById("region").value;
    const source = document.getElementById("source").value;
    const type = document.getElementById("type").value;
    const translated = document.getElementById("translated").value;

    passages.forEach((passage) => {
      try {
        const metadata = JSON.parse(passage.dataset.metadata);

        // Check if the passage matches all selected filters
        const matchesRegion = !region || metadata.world_bank_region === region;
        const matchesSource = !source || metadata.source === source;

        // Handle types array
        let matchesType = true;
        if (type) {
          const typesList = metadata.types
            ? metadata.types.replace(/^\['|'\]$/g, "").split("', '")
            : [];
          matchesType = typesList.includes(type);
        }

        const matchesTranslated =
          !translated || metadata.translated === translated;

        // Show/hide the passage based on all filters
        const visible =
          matchesRegion && matchesSource && matchesType && matchesTranslated;
        passage.style.display = visible ? "block" : "none";
      } catch (e) {
        console.error("Error applying filters to passage:", e);
        passage.style.display = "none"; // Hide passages with invalid metadata
      }
    });

    updateVisibleCount();
  }

  // Shuffle functionality
  document.getElementById("shuffle").addEventListener("click", () => {
    const shuffled = passages
      .map((value) => ({ value, sort: Math.random() }))
      .sort((a, b) => a.sort - b.sort)
      .map(({ value }) => value);

    shuffled.forEach((passage) => passagesContainer.appendChild(passage));
  });

  // Sort by length functionality
  document.getElementById("sortLength").addEventListener("click", () => {
    const sorted = passages.sort(
      (a, b) => parseInt(b.dataset.length) - parseInt(a.dataset.length)
    );

    sorted.forEach((passage) => passagesContainer.appendChild(passage));
  });

  // Update visible count
  function updateVisibleCount() {
    const visiblePassages = passages.filter(
      (p) => p.style.display !== "none" && !p.classList.contains('search-hidden')
    ).length;
    visibleCountSpan.textContent = visiblePassages;
  }

  // Add event listeners to filters
  ["region", "source", "type", "translated"].forEach((id) => {
    document.getElementById(id).addEventListener("change", applyFilters);
  });

  // Add search functionality
  function handleSearch() {
    const searchText = searchInput.value.trim();
    let regex;

    // Remove existing search highlights
    passages.forEach(passage => {
      const textElement = passage.querySelector('.passage-text');
      textElement.innerHTML = textElement.innerHTML.replace(/<mark class="search-match">(.*?)<\/mark>/g, '$1');
    });

    // Check if input is a regex pattern
    const regexMatch = searchText.match(/^\/(.+)\/([gi]*)$/);
    
    try {
      if (regexMatch) {
        // If it's a regex pattern, create RegExp object
        regex = new RegExp(regexMatch[1], regexMatch[2]);
      } else if (searchText) {
        // If it's plain text, create case-insensitive regex
        regex = new RegExp(searchText, 'i');
      }
    } catch (e) {
      console.warn('Invalid regex pattern:', e);
      return;
    }

    passages.forEach((passage) => {
      const textElement = passage.querySelector('.passage-text');
      const passageText = textElement.textContent;
      
      if (!searchText) {
        // If search is empty, don't hide based on search
        passage.classList.remove('search-hidden');
      } else {
        // Test for match first
        const matches = regex ? regex.test(passageText) : false;
        passage.classList.toggle('search-hidden', !matches);

        // If visible, highlight matches
        if (matches) {
          // Create a global version of the regex for highlighting all matches
          const globalRegex = regexMatch 
            ? new RegExp(regexMatch[1], regexMatch[2].includes('g') ? regexMatch[2] : regexMatch[2] + 'g')
            : new RegExp(searchText, 'ig');

          // Get HTML content and wrap matches in mark tags while preserving existing HTML
          let html = textElement.innerHTML;
          const textNodes = [];
          const walk = document.createTreeWalker(
            textElement,
            NodeFilter.SHOW_TEXT,
            null,
            false
          );

          let node;
          while (node = walk.nextNode()) {
            textNodes.push(node);
          }

          // Process in reverse order to not mess up node positions
          for (let i = textNodes.length - 1; i >= 0; i--) {
            const node = textNodes[i];
            const text = node.textContent;
            const highlightedText = text.replace(
              globalRegex,
              match => `<mark class="search-match">${match}</mark>`
            );
            
            if (text !== highlightedText) {
              const span = document.createElement('span');
              span.innerHTML = highlightedText;
              node.parentNode.replaceChild(span, node);
            }
          }
        }
      }
    });

    updateVisibleCount();
  }

  // Add search input event listener
  searchInput.addEventListener('input', handleSearch);

  // Initialize
  populateFilters();
});
