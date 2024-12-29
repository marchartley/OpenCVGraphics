import numpy as np

from Graphics import Graphic, SceneRender
import cv2

def main():
    EPSILON = 1
    WIDTH, HEIGHT = 800, 600

    # On crée un élément graphique (un joli smiley)
    # cv2.IMREAD_UNCHANGED permet de conserver la transparence
    smiley = Graphic(cv2.imread("smiley.png", cv2.IMREAD_UNCHANGED))
    smiley.resize((50, 50), cv2.INTER_NEAREST)

    # Utilisation de la webcam
    cap = cv2.VideoCapture(0)

    # Instancie le "moteur de rendu"
    render = SceneRender((WIDTH, HEIGHT))


    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        # On transforme "img" en un élément graphique manipulable
        webcam = Graphic(img)
        # webcam.fill((0, 0, 0))

        # On réduit l'image de la webcam
        webcam.resize((100, 100))
        # Ajout d'un petit effet visuel sur la camera
        webcam.apply_sketch_effect()
        # Et d'un peu de texte
        webcam.draw_text('Hello, OpenCV!', (50, 250), 'Fonts/Hollster.ttf', 30, (0, 0, 0), alpha = 0.8)

        # On crée un "tableau blanc"
        caneva = Graphic((WIDTH, HEIGHT))
        # On le rempli en blanc
        caneva.fill((255, 255, 255))

        # On dessine un rectangle de (10, 30) à (220, 170) bleu, rempli (thickness = -1)
        caneva.draw_rectangle((10, 30), (220, 170), (255, 100, 0), -1)
        # On y dessine un cercle centré en (90, 100), de 50 pixels de rayon, rouge, rempli (-1), et avec une légère transparence
        caneva.draw_circle((90, 100), 50, (0, 0, 255), -1, alpha=0.9)
        # Un deuxième cercle centré en (100, 50), de 50 pixels de rayon, jaune, rempli (-1), et avec une plus forte transparence
        caneva.draw_circle((140, 100), 50, (0, 255, 255), -1, alpha=0.5)

        mask = np.zeros((HEIGHT, WIDTH))
        for i in range(HEIGHT):
            for j in range(WIDTH):
                mask[i, j] = abs(10 - (i + j) % 20) / 10.0 if 50 < i < 100 else 0
        caneva.add_mask(mask, use_values_between_0_and_1=True)


        render.clear()
        # On place en fond le caneva
        render.add_layer(caneva)
        # Puis au premier plan la webcam (en bas à droite)
        render.add_layer(webcam, (WIDTH - 100, HEIGHT - 100))
        # Encore au dessus, un petit smiley
        render.add_layer(smiley, (WIDTH - 125, HEIGHT - 125), 0.5)

        output = render.get_image()
        cv2.imshow("Resultat", output)

        key = cv2.waitKey(EPSILON) & 0xFF
        if key == ord("q") or key == 27:
            break


        if key != 0xFF:
            if key == ord("1"):
                print("La touche '1' est appuyée")
                # Faire des actions
            if key == ord("2"):
                print("La touche '2' est appuyée")
                # Faire des actions

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()