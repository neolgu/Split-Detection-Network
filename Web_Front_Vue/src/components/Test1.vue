<template>
  <div align="center">

    <div class="dropbox" id="dropbox">
      <input class="image_select" type="file" @change="onImageSelected" >
      <p>
        Drag your file(s) here to begin or click to browse
      </p>

    </div>

    <br>

    <div class="selected_image_preview">
      <cropper ref="cropper" :src="selectedImage" @change="change"></cropper>
    </div>

    <br>

    <div class="cropped_image_preview">
      <img style="width:100%; height:100%;" :src="croppedImage"/>
    </div>

    <br>

    <div>
      <button class="cropped_image_test" @click="testImage">Test</button>
    </div>
  </div>
</template>

<script>

import { Cropper } from 'vue-advanced-cropper';
import 'vue-advanced-cropper/dist/style.css';

export default {
  name: "test1",
  components: {
    Cropper,
  },
  data () {
    return {
      selectedImage: require('../assets/default_image.jpg'),
      croppedImage: require('../assets/default_image.jpg')
    }
  },
  methods: {
    onImageSelected(event) {
      let image = event.target.files[0];
      if(image==null) {
        this.selectedImage = require('../assets/default_image.jpg');
        return;
      }
      let reader = new FileReader();
      reader.readAsDataURL(image);
      reader.onload = event => {
        this.selectedImage = event.target.result;
      }
      document.getElementById("dropbox").style.display = "none"
    },
    change() {
      const result = this.$refs.cropper.getResult();
      this.croppedImage = result.canvas.toDataURL();
    },
    //사진을 서버로 전송!
    testImage() {
      const frm = new FormData();
      frm.append('img', this.croppedImage);
      this.$axios.post("http://34.92.225.82:5001/predict", frm)
          .then(res => {
            if(res.data.pred_result===1) {
              const output = "This image is probably a 'REAL' image.\n" + "Output score of our detector: " + res.data.pred_probability.toFixed(2);
              alert(output);
            } else {
              const output = "This image is probably a 'FAKE' image.\n" + "Output score of our detector: " + res.data.pred_probability.toFixed(2);
              alert(output);
            }
          });
    }
  }
};
</script>

<style>
.dropbox {
  outline: 2px dashed grey; /* the dash box */
  outline-offset: -10px;
  background: lightcyan;
  color: dimgray;
  padding: 10px 10px;
  min-height: 400px; /* minimum height */
  min-width: 20px;
  position: relative;
  cursor: pointer;
  width: 800px;
  height: 200px;
}
.image_select {
  opacity: 0; /* invisible but it's there! */
  width: 600px;
  height: 200px;
  position: absolute;
  cursor: pointer;
}

.dropbox:hover {
  background: lightblue; /* when mouse over to the drop zone, change color */
}

.dropbox p {
  font-size: 1.2em;
  text-align: center;
  padding: 50px 0;
}
.image_select{
  border: 3px solid aquamarine;
  cursor: pointer;
}
.selected_image_preview{
  border: 3px solid aquamarine;
  width: 50%;
  height: 100%;
}
.cropped_image_preview{
  border: 3px solid aquamarine;
  width: 200px;
  height: 200px;
}


.cropped_image_test {
  display: inline-block;
  border-radius: 4px;
  background-color: #3eb489;
  border: none;
  color: #FFFFFF;
  text-align: center;
  font-size: 18px;
  padding: 20px;
  width: 200px;
  transition: all 0.5s;
  cursor: pointer;
  margin: 5px;
}

.cropped_image_test span {
  cursor: pointer;
  display: inline-block;
  position: relative;
  transition: 0.5s;
}

.cropped_image_test span:after {
  content: '\00bb';
  position: absolute;
  opacity: 0;
  top: 0;
  right: -20px;
  transition: 0.5s;
}

.cropped_image_test:hover span {
  padding-right: 25px;
}

.cropped_image_test:hover span:after {
  opacity: 1;
  right: 0;
}
</style>