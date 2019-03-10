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

var buildOutput = function(resp, num_epochs) {
    var output = document.getElementById(num_epochs);
    var data = resp[num_epochs];
    output.innerHTML = data['confidence'] + ' sure<br> that you drew<br> the number ' + data['prediction'];
};

var requestPrediction = function() {
    var endpoint = 'http://192.168.188.110:8081/api/predict';
    var gsarr = getGrayscaleArray();
    var b64encoded = btoa(String.fromCharCode.apply(null, gsarr));
    var data = {
        'b64img': b64encoded
    };
    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
            var resp = JSON.parse(xhr.response);
            console.log('Predictions:', resp);
            buildOutput(resp, '10_epochs');
            buildOutput(resp, '100_epochs');
            buildOutput(resp, '1000_epochs');
            // resp == { 'prediction': X, 'confidence': Y }
            // var bestguess = resp['1000_epochs']
            // alert('skynet is ' + bestguess['confidence'] + ' sure that you drew the number ' + bestguess['prediction']);
        }
    }
    xhr.open('POST', endpoint, true);
    xhr.setRequestHeader('Content-Type', 'application/json; charset=UTF-8');
    var payload = JSON.stringify(data);
    xhr.send(payload);
};
