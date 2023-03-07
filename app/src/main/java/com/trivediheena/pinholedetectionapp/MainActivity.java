package com.trivediheena.pinholedetectionapp;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import android.Manifest;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Size;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.google.common.util.concurrent.ListenableFuture;
import com.trivediheena.pinholedetectionapp.ml.TilesModel;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.image.TensorImage;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    ImageCapture imageCapture;
    ImageAnalysis imageAnalysis;
    ImageView imgPredict;
    Button btnSelect,btnPredict;
    TextView txtPred;
    Bitmap bitmap=null;
    String ans="Hello World!!!";
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ActivityCompat.requestPermissions(MainActivity.this,new String[]{Manifest.permission.CAMERA},1);
        PreviewView viewFinder = findViewById(R.id.view_finder);
        //Camera camera = cameraProvider.bindToLifecycle(lifecycleOwner, cameraSelector, preview);
        imgPredict=findViewById(R.id.imgPredict);
        btnSelect=findViewById(R.id.btnImage);
        btnPredict=findViewById(R.id.btnPredict);
        txtPred=findViewById(R.id.txtPredicted);
        Preview preview = new Preview.Builder().build();
        ListenableFuture cameraProviderFuture =
                ProcessCameraProvider.getInstance(getApplicationContext());

        cameraProviderFuture.addListener(() -> {
            try {
                // Camera provider is now guaranteed to be available
                ProcessCameraProvider cameraProvider = (ProcessCameraProvider) cameraProviderFuture.get();

                // Set up the view finder use case to display camera preview
                //Preview preview = new Preview.Builder().build();

                // Set up the capture use case to allow users to take photos
                imageCapture = new ImageCapture.Builder()
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                        .build();

                // Choose the camera by requiring a lens facing
                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                        .build();

                // Attach use cases to the camera with the same lifecycle owner
/*                        Camera camera = cameraProvider.bindToLifecycle(
                                ((LifecycleOwner) MainActivity.this),
                                cameraSelector,
                                imageAnalysis,
                                preview,
                                imageCapture);*/

                imageAnalysis=new ImageAnalysis.Builder()
                        .setTargetResolution(new Size(640, 480))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

                imageAnalysis.setAnalyzer(Executors.newSingleThreadExecutor(), new ImageAnalysis.Analyzer() {
                    @Override
                    public void analyze(@NonNull ImageProxy image) {
                        bitmap=imageProxyToBitmap(image);
                    }
                });
                cameraProvider.bindToLifecycle((LifecycleOwner)MainActivity.this,cameraSelector,imageAnalysis,preview);
                // Connect the preview use case to the previewView
                preview.setSurfaceProvider(
                        viewFinder.getSurfaceProvider());
            } catch (InterruptedException | ExecutionException e) {
                // Currently no exceptions thrown. cameraProviderFuture.get()
                // shouldn't block since the listener is being called, so no need to
                // handle InterruptedException.
            }
        }, ContextCompat.getMainExecutor(getApplicationContext()));

        btnSelect.setOnClickListener(new View.OnClickListener(){
           @Override
           public void onClick(View v){
               Intent intent=new Intent(Intent.ACTION_GET_CONTENT);
               intent.setType("image/*");
               startActivityForResult(intent,100);
           }
        });
        btnPredict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        txtPred.setText(classify(bitmap)+"");
                    }
                });
            }
        });
    }

    public String classify(Bitmap bmp){
        bitmap=Bitmap.createScaledBitmap(bmp,120,120,true);
        try {
            TilesModel model = TilesModel.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 120, 120, 3}, DataType.FLOAT32);
            TensorImage tensorImage=new TensorImage(DataType.FLOAT32);
            tensorImage.load(bitmap);
            ByteBuffer byteBuffer=tensorImage.getBuffer();
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            TilesModel.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            // Releases model resources if no longer used.
            model.close();
            if(outputFeature0.getFloatArray()[0]>outputFeature0.getFloatArray()[1])
                ans="Pinhole"+"\n["+outputFeature0.getFloatArray()[0]+","+outputFeature0.getFloatArray()[1]+"]";//txtPred.setText("Pinhole");
            else if(outputFeature0.getFloatArray()[0]<outputFeature0.getFloatArray()[1])
                ans="No Pinhole"+"\n["+outputFeature0.getFloatArray()[0]+","+outputFeature0.getFloatArray()[1]+"]";//txtPred.setText("No Pinhole");
        } catch (IOException e) {
            // TODO Handle the exception
        }
        return ans;
    }
    private Bitmap imageProxyToBitmap(ImageProxy image) {
        ByteBuffer buffer = ByteBuffer.allocate(image.getHeight()*image.getWidth()*2); //image.getPlanes()[0].getBuffer();
        ByteBuffer y = image.getPlanes()[0].getBuffer();
        ByteBuffer cr = image.getPlanes()[1].getBuffer();
        ByteBuffer cb = image.getPlanes()[2].getBuffer();
        buffer.put(y);
        buffer.put(cb);
        buffer.put(cr);
        buffer.rewind();
        YuvImage yuvImage = new YuvImage(buffer.array(),
                ImageFormat.NV21, image.getWidth(), image.getHeight(), null);

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0,
                image.getWidth(), image.getHeight()), 50, out);
        byte[] bytes = out.toByteArray();            //new byte[buffer.capacity()];
        //buffer.get(bytes);
        byte[] clonedBytes = bytes.clone();
        Bitmap bmp=BitmapFactory.decodeByteArray(clonedBytes,0,clonedBytes.length);
        return bmp;
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        switch (requestCode){
            case 100:
                Uri uri=data.getData();
                imgPredict.setImageURI(uri);
                try{
                    bitmap= MediaStore.Images.Media.getBitmap(this.getContentResolver(),uri);
                    //String str="Answer";
                }catch (IOException ie){
                    ie.printStackTrace();
                }

        }
    }
}