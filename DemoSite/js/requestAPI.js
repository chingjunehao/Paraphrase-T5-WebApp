var old_val = "";
function passingToAPI() {

  var xhr = new XMLHttpRequest();
  var url = "http://127.0.0.1:5000/paraphrase-api";
  xhr.open("POST", url, true);
  xhr.setRequestHeader("Content-Type", "application/json");

  var x = document.getElementById("texttbt");
  var data = JSON.stringify({"text": x.value});
  xhr.send(data);
  
  xhr.onreadystatechange = function () {
    // while waiting for the result to be returned
      if (xhr.readyState === 3){ 
        var waiting = old_val.concat("...")
        document.getElementById("output").innerHTML = waiting;
      }

      if (xhr.readyState === 4 && xhr.status === 200) {
          document.getElementById("output").innerHTML = xhr.responseText;
          old_val = xhr.responseText;
      }

      
  };

    
    
}