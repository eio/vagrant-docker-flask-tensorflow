var getGrayscaleArray = function() {
    var canvas = document.getElementById('mnist');
    var ctx = canvas.getContext('2d');
    var data = ctx.getImageData(0,0,28,28).data;
    var gsarr = [];
    for (var i = 0; i < data.length; i += 4) {
        // // RGB values are always 0
        // var r = data[i];
        // var g = data[i + 1];
        // var b = data[i + 2];
        var a = data[i + 3];
        var scaled = (a/255);
        var val = 0;
        if(scaled > 0){
            val = 1;
        }
        gsarr.push(val);
    }
    return new Uint8Array(gsarr);
};

var requestPrediction = function() {
    var endpoint = 'http://127.0.0.1:8081/api/predict';
    var gsarr = getGrayscaleArray();
    // console.log(String(gsarr))
    var b64encoded = btoa(String.fromCharCode.apply(null, gsarr));
    var data = {
        'b64img': b64encoded
    };
    console.log(data);
    // construct an HTTP request
    var xhr = new XMLHttpRequest();
    // callback on response
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
            console.log(xhr.response)
            var resp = JSON.parse(xhr.response);
            console.log(resp)
            alert('Prediction:', resp.prediction);
        }
    }
    xhr.open('POST', endpoint, true);
    xhr.setRequestHeader('Content-Type', 'application/json; charset=UTF-8');
    // send the collected data as JSON
    var payload = JSON.stringify(data);
    xhr.send(payload);
};
