document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('search');
    const tableRows = document.querySelectorAll('tbody tr');

    function searchConcepts(searchTerm) {
        searchTerm = searchTerm.toLowerCase();
        
        tableRows.forEach(row => {
            const conceptCell = row.querySelector('td:first-child');
            const conceptText = conceptCell.textContent.trim().toLowerCase();
            const isMatch = conceptText.includes(searchTerm);
            
            row.classList.toggle('search-hidden', !isMatch);
        });
    }

    searchInput.addEventListener('input', (e) => {
        searchConcepts(e.target.value);
    });
}); 
