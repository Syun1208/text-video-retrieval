const formEl = document.querySelector("form")
const inputEl = document.getElementById("search-input")
const searchResults = document.querySelector(".search-results")
const showMore = document.getElementById("show-more-button")

var loc = window.location.pathname
var dir = loc.substring(0, loc.lastIndexOf('/'))
var path = require('path');

let inputData = ""
let page = 1;

async function textSearch(){
    inputData = inputEl.value;
    const url = 'http://0.0.0.0:8090/text_search'
    const data = {
        'text': inputData,
        'image_id': inputData,
        "image_base64": '',
        "base64_optional": false,
        "k": 12
    }

    const response = await fetch(url, {
        method: "POST", 
        headers: {
          "Content-Type": "application/json",
          "accept": "application/json",
        },
        body: JSON.stringify(data), 
      });
      
      const response_json = await response.json()
      const results = response_json.results

      if (page == 1) {
        searchResults.innerHTML = ""
      }

      results.map((result) =>{
        const imageWrapper = document.createElement('div')
        imageWrapper.classList.add("search-result")
        const image = document.createElement("img")
        image.src = path.join(dir, result.image_paths)
        image.alt = 'Path: ' + result.image_paths + 'Score: ' + result.scores
        const imageLink = document.createElement('a')
        imageLink.href = path.join(dir, result.image_paths);
        imageLink.target = "_blank";
        imageLink.textContent = 'Path: ' + result.image_paths + 'Score: ' + result.scores;

        imageWrapper.appendChild(image);
        imageWrapper.appendChild(imageLink);
        imageWrapper.appendChild(imageWrapper);
      });

      page++
      if (page > 1){
        showMore.style.display = "block"
      }
}

formEl.addEventListener("submit", (event) => {
    event.preventDefault()
    page=1
    textSearch()
  })


showMore.addEventListener("submit", (event) => {
    event.preventDefault()
    page=1
    textSearch()
})