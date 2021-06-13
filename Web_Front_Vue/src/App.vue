<template>
  <div>

    <div align="left" class="head_2">

    </div>

    <div align="center">
      <div class="line">
        <img src="../public/logo.png" width="50px" height="50px">
      </div>
      <div class="line">
        <h1 align="center" style="color:green; font-family:cursive;">Welcome to Deepfake Detection !</h1>
      </div>
      <br>
      <div>
        <details class="detail_1">
          <summary style="font-weight: bold; color: green">Guide</summary>
          <ul align="left">
            <li class="note">1. Select one of 2 options("Test for Image" or "Test for Video")</li>
            <li class="note">2-1. For "Test for Image", drag and drop or browse an image you want</li>
            <li class="note">2-2. For "Test for Video", drag and drop or browse a video you want and select a frame you want using progress bar</li>
            <li class="note">3. Crop the image(frame) (tip - Cropping only the face part would lead to better outcome)</li>
            <li class="note">4. Submit the cropped image(frame) to server</li>
            <li class="note">5. After a few seconds, the outcome that notifies that the cropped image(frame) is real or fake would be alerted!</li>
            <li style="color: orangered">NOTE : Any images or videos you upload are note stored in our server as part of copyright protection</li>
          </ul>
        </details>
      </div>
      <br>
      <div align="center">
        <button class="btn" style="vertical-align:middle" @click="imageTest"><span>Test for Image</span></button>
        <button class="btn" style="vertical-align:middle" @click="videoTest"><span>Test for Video</span></button>
      </div>

      <br>

      <div align="center">
        <router-view>
        </router-view>
      </div>

      <br>
      <br>

      <div align="center" >
        <img src="./assets/Sign_Logo.png" class="head_1">
      </div>
    </div>
  </div>
</template>

<script>

export default {
  name: 'Home',
  components: {},
  methods: {

    imageTest: function () {
      this.$router.push('/test1');
    },

    videoTest: function () {
      this.$router.push('/test2');
    }
  }
}

</script>

<style>

.btn {
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

.btn span {
  cursor: pointer;
  display: inline-block;
  position: relative;
  transition: 0.5s;
}

.btn span:after {
  content: '\00bb';
  position: absolute;
  opacity: 0;
  top: 0;
  right: -20px;
  transition: 0.5s;
}

.btn:hover span {
  padding-right: 25px;
}

.btn:hover span:after {
  opacity: 1;
  right: 0;
}
.line{
  display: inline-block;
}
.head_1 {
  display:inline-block;
}
.head_2 {
  display:inline-block;
  float: bottom;
}
.detail_1{
  width: 750px;
}
.note{
  color: darkturquoise;
}
</style>