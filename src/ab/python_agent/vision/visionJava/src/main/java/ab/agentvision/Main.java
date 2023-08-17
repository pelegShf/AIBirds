package ab.agentvision;

import ab.vision.ABObject;
import ab.vision.Vision;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.Rectangle;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        try  (InputStream inputStream = Main.class.getResourceAsStream("/screenshot.jpg")){
            // Load the image file from the same folder
            if (inputStream != null) {
                BufferedImage image = ImageIO.read(inputStream);
                Vision vis = new Vision(image);
                List<ABObject> sling = vis.findBirdsMBR();
                System.out.println(sling.get(0).x);
                System.out.println(sling.get(0).x);
            } else {
                System.err.println("Image not found");
            }


            // Now 'image' contains your loaded image as a BufferedImage
            // You can use 'image' as needed, e.g., displaying or processing it
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}