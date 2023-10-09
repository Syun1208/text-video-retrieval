const formEl = document.querySelector("form")
const inputData = document.getElementById("search-input")
const searchResults = document.querySelector(".search-results")
const showMore = document.getElementById("show-more-button")
const imageId = document.getElementById("image-id")
const searchButton = document.getElementById("search-button")

var loc = window.location.pathname
var dir = loc.substring(0, loc.lastIndexOf('/'))
const baseDir = '/media/hoangtv/New Volume/backup/'

let page = 1;




// Get a reference to the button element by its ID
var addButton = document.getElementById('search-button');

// Add a click event listener to the button
addButton.addEventListener('click', textSearch);
async function textSearch() {
  ///  Example Curl 
  //   curl -X 'POST' \
  //   'http://localhost:8090/text_search' \
  //   -H 'accept: application/json' \
  //   -H 'Content-Type: application/json' \
  //   -d '{
  //     "text": "Đoạn video trình chiếu 3 chiếc điện thoại Samsung trong buổi ra mắt sản phẩm. Ban đầu, từng chiếc điện thoại hiện lên lần lượt và cảnh cả 3 chiếc điện thoại cùng xuất hiện",
  //     "image_id": 1,
  //     "image_path": "afsafaf",
  //     "k": 50,
  //     "mode_search": "blip",
  //     "base64_optional": false,
  //     "submit_name": "query-1"
  // }

  const url = 'http://localhost:8090/text_search';

  const headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
  };
  const data = {
    text: inputData.value,
    image_id: Number(imageId.value),
    image_path: "afsafaf",
    k: 50,
    mode_search: "blip",
    base64_optional: true,
    submit_name: "query-1"
  };
  const requestOptions = {
    method: 'POST',
    headers: headers,
    body: JSON.stringify(data)
  };
  const response = await fetch(url, requestOptions)
  const bodyData = await response.json()
  console.log(bodyData)
  bodyData.images_base64.map((data, indx) => {
    const getSearch = document.getElementById("search-results")
    var newElement = document.createElement('p');

    // Set some content for the new element
    newElement.textContent = 'New element added!';
    console.log(data)
    getSearch.appendChild(newElement)
  })
 
}