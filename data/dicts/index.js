const express = require('express');
const app = express();
const port = 3123;
const fs = require('fs');
const cors = require('cors');
app.use(cors({
    origin: '*',
    methods: ['GET', 'POST', 'PUT', 'DELETE'],
    allowedHeaders: ['Content-Type', 'Authorization']
}));
// Read and parse mapData.json only once
const jsonData = JSON.parse(fs.readFileSync('mapData.json', 'utf8'));

const convertStringToRegexp = (text) => {
    const normalizedText = text
        .normalize('NFD')
        .replace(/[\u0300-\u036f]/g, '') // Remove all accents
        .replace(/[|\\{}()[\]^$+*?.]/g, '\\$&') // Remove all regex reserved characters
        .toLowerCase();

    // Use a map for replacements to avoid repetition
    const replacements = {
        'a': '[a,á,à,ả,ã,ạ,â,ấ,ầ,ẩ,ẫ,ậ,ă,ắ,ằ,ẳ,ẵ,ặ]',
        'e': '[e,é,è,ê,ẻ,ể,ẽ,ế,ề,ệ,ễ,ẹ]',
        'd': '[d,đ]',
        'y': '[y,ý,ỳ,ỷ,ỹ,ỵ]',
        'i': '[i,í,ì,î,ỉ,ị,ĩ]',
        'o': '[o,ó,ò,õ,ô,ố,ồ,ổ,ộ,ỗ,ớ,ờ,ợ,ở,ỡ,ơ]',
        'u': '[u,ü,ú,ù,û,ủ,ụ,ũ,ư,ứ,ừ,ự,ử,ữ]'
    };

    let regexp = `.*${normalizedText}.*`;
    for (let key in replacements) {
        regexp = regexp.replace(new RegExp(key, 'g'), replacements[key]);
    }

    return new RegExp(regexp, 'i');
};

app.get('/:keyword', (req, res) => {
    const keyword = req.params.keyword;
    try {
        const regexKeyword = convertStringToRegexp(keyword);
        const key = Object.keys(jsonData).find((key) => regexKeyword.test(key));
        
        if (!key) {
            return res.status(404).send("Not found");
        }

        res.send(jsonData[key]);
    } catch (error) {
        console.error(error);
        res.status(500).send("Internal server error");
    }
});

app.get('/test/convertToMap', (req, res) => {
    try {
        const mapData = new Map();
        jsonData.forEach((item)=>{
            mapData.set(item.text,item);
        })
        const mapDataJson = JSON.stringify(Object.fromEntries(mapData));
        fs.writeFileSync('mapData.json', mapDataJson);
        res.send("success");
    } catch (error) {
        console.error(error);
        res.status(500).send("Internal server error");
    }
});

app.listen(port, () => console.log(`Example app listenin22g on 1port ${port}!`));
