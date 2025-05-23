var ocrDemo = {
    HOST: "http://localhost",
    PORT: "8080",
    
    CANVAS_WIDTH: 196,
    TRANSLATED_WIDTH: 28,
    PIXEL_WIDTH: 7, // TRANSLATED_WIDTH = CANVAS_WIDTH / PIXEL_WIDTH

    onLoadFunction: function () {
        const canvas = document.getElementById("canvas");
        const ctx = canvas.getContext("2d");

        this.ctx = ctx;
        this.canvas = canvas;
        this.data = Array(this.TRANSLATED_WIDTH * this.TRANSLATED_WIDTH).fill(0);
        this.trainArray = [];
        this.trainingRequestCount = 0;
        this.BATCH_SIZE = 1;
        this.BLACK = "#000000";

        this.drawGrid(ctx);

        canvas.addEventListener("mousedown", (e) => this.onMouseDown(e,ctx, canvas));
        canvas.addEventListener("mousemove", (e) => this.onMouseMove(e, ctx, canvas));
        canvas.addEventListener("mouseup", (e) => this.onMouseUp(e));

    },

    drawGrid: function(ctx) {
        for (var x = this.PIXEL_WIDTH, y = this.PIXEL_WIDTH; 
                 x < this.CANVAS_WIDTH; x += this.PIXEL_WIDTH, 
                 y += this.PIXEL_WIDTH) {
            ctx.strokeStyle = this.BLACK;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, this.CANVAS_WIDTH);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(this.CANVAS_WIDTH, y);
            ctx.stroke();
        }
    },
    onMouseMove: function(e, ctx, canvas) {
        if (!canvas.isDrawing) {
            return;
        }
        var rect = canvas.getBoundingClientRect();
        var x = e.clientX - rect.left;
        var y = e.clientY - rect.top;

        console.log(`Mouse Move at: ${x}, ${y}`);

        this.fillSquare(ctx, x, y);
    },

    onMouseDown: function(e, ctx, canvas) {
        canvas.isDrawing = true;

        var rect = canvas.getBoundingClientRect();
        var x = e.clientX - rect.left;
        var y = e.clientY - rect.top;

        console.log(`Mouse Down at: ${x}, ${y}`);

        this.fillSquare(ctx, x, y);
    },

    onMouseUp: function(e) {
        this.canvas.isDrawing = false;
    },

    fillSquare: function(ctx, x, y) {
        var xPixel = Math.floor(x / this.PIXEL_WIDTH);
        var yPixel = Math.floor(y / this.PIXEL_WIDTH);
        var index = yPixel * this.TRANSLATED_WIDTH + xPixel;

        if (index >= 0 && index < this.data.length) {
            // Accumulate intensity (max 1)
            this.data[index] = Math.min(1, this.data[index] + 0.1);

            // Convert to grayscale (0 = black, 1 = white)
            let gray = Math.floor((1 - this.data[index]) * 255);
            ctx.fillStyle = `rgb(${gray}, ${gray}, ${gray})`;
            ctx.fillRect(
                xPixel * this.PIXEL_WIDTH,
                yPixel * this.PIXEL_WIDTH,
                this.PIXEL_WIDTH,
                this.PIXEL_WIDTH
            );
        }
    },


    resetCanvas: function() {
        this.data = Array(this.TRANSLATED_WIDTH * this.TRANSLATED_WIDTH).fill(0);
        const ctx = this.ctx;
        const canvas = this.canvas;
        ctx.clearRect(0, 0, canvas.width, canvas.height);  
        this.drawGrid(ctx);  
    },
    train: function() {
        var digitVal = document.getElementById("digit").value;
        if (!digitVal || this.data.indexOf(1) < 0) {
            alert("Please type and draw a digit value in order to train the network");
            return;
        }
        this.trainArray.push({"y0": this.data, "label": parseInt(digitVal)});
        this.trainingRequestCount++;

        // Time to send a training batch to the server
        if (this.trainingRequestCount == this.BATCH_SIZE) {
            alert("Sending training data to server...");
            var json = {
                trainArray: this.trainArray,
                train: true
            };

            this.sendData(json);
            this.trainingRequestCount = 0;
            this.trainArray = [];
        }
    },
    test: function() {
        if (this.data.indexOf(1) < 0) {
            alert("Please draw a digit in order to test the network");
            return;
        }
        var json = {
            image: this.data,
            predict: true
        };
        this.sendData(json);
    },
    receiveResponse: function(xmlHttp) {
        if (xmlHttp.status != 200) {
            alert("Server returned status " + xmlHttp.status);
            return;
        }
        var responseJSON = JSON.parse(xmlHttp.responseText);
        if (xmlHttp.responseText && responseJSON.type == "test") {
            alert("The neural network predicts you wrote a \'" 
                   + responseJSON.result + '\'');
        }
    },

    onError: function(e) {
        alert("Error occurred while connecting to server: " + e.target.statusText);
    },

    sendData: function(json) {
        var xmlHttp = new XMLHttpRequest();
        xmlHttp.open('POST', this.HOST + ":" + this.PORT, true); 

        xmlHttp.setRequestHeader('Content-Type', 'application/json');

        xmlHttp.onload = function() {
            this.receiveResponse(xmlHttp);
        }.bind(this);

        xmlHttp.onerror = function() {
            this.onError(xmlHttp);
        }.bind(this);
    
        var msg = JSON.stringify(json);
        xmlHttp.send(msg); 
    }

};